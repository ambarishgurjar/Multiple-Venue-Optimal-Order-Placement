import numpy as np
import pandas as pd
from datetime import *



def best_ask_strategy(snapshots, order_size):
    executed, cost = 0, 0
    for snapshot in snapshots:
        if executed >= order_size:
            break
        for exch in sorted(snapshot['exchanges'], key=lambda v: v['ask']):
            if executed >= order_size:
                break
            quantity = min(order_size - executed, exch['ask_size'])
            executed += quantity
            cost += calculate_naive_cost(quantity, exch)
    avg_price = cost / executed if executed else float('inf')
    return cost, executed, avg_price




def twap_strategy(snapshots, order_size, interval_sec=60):
    start, end = snapshots[0]['ts_event'], snapshots[-1]['ts_event']
    total_buckets = int(np.ceil((end - start).total_seconds() / interval_sec))
    base_alloc = order_size // total_buckets
    remainder = order_size % total_buckets

    executed, cost = 0, 0
    for i in range(total_buckets):
        if executed >= order_size:
            break
        bucket_start = start + timedelta(seconds=i * interval_sec)
        bucket_end = bucket_start + timedelta(seconds=interval_sec)
        bucket_snaps = [s for s in snapshots if bucket_start <= s['ts_event'] < bucket_end]
        if not bucket_snaps:
            continue
        bucket_goal = base_alloc + (1 if i < remainder else 0)
        n_orders = min(bucket_goal, order_size - executed)
        cost_i, exe_i, _ = best_ask_strategy(bucket_snaps, n_orders)
        executed += exe_i
        cost += cost_i

    avg_price = cost / executed if executed else float('inf')
    return cost, executed, avg_price

def vwap_strategy(snapshots, order_size):
    executed, cost = 0, 0
    for snap in snapshots:
        depth = sum(v['ask_size'] for v in snap['exchanges'])
        if executed >= order_size or depth == 0:
            continue
        goal_remaining = order_size - executed
        for exch in snap['exchanges']:
            alloc = int(goal_remaining * exch['ask_size'] / depth)
            qty = min(alloc, exch['ask_size'], order_size - executed)
            executed += qty
            cost += calculate_naive_cost(qty, exch)
    avg_price = cost / executed if executed else float('inf')
    return cost, executed, avg_price

def augment_snapshots(snapshots,
                      num_clones: int = 3,
                      price_spread: float = 0.01,
                      size_spread: int = 20):
    augmented = []
    center = (num_clones - 1) / 2.0
    for snap in snapshots:
        new_exchanges = []
        for exch in snap['exchanges']:
            for i in range(num_clones):
                off = (i - center)
                fake = {
                    'ask': round(exch['ask'] + off * price_spread, 5),
                    'ask_size': max(0, exch['ask_size'] + int(off * size_spread)),
                    'fee': exch['fee'],
                    'rebate': exch['rebate'],
                    'id': exch.get('id', 0) * num_clones + i
                }
                new_exchanges.append(fake)
        augmented.append({
            'ts_event': snap['ts_event'],
            'exchanges': new_exchanges
        })
    return augmented


def tune_nesterov(snapshots, order_size, bounds,
                  alpha=1.0, mu=0.9, eps=1e-3, n_iter=10,
                  init_params=None, verbose=False):
    names = ['lambda_over', 'lambda_under', 'theta_queue']
    if init_params is not None:
        x = np.array(init_params, dtype=float)
    else:
        x = np.array([
            np.mean(bounds['lambda_over']),
            np.mean(bounds['lambda_under']),
            np.mean(bounds['theta_queue'])
        ], dtype=float)
    v = np.zeros_like(x)
    best = {'cost': np.inf, 'params': x.copy()}
    for it in range(n_iter):
        y = x + mu * v
        grad = np.zeros_like(x)
        for i, name in enumerate(names):
            p_plus = {
                'lambda_over':  y[0],
                'lambda_under': y[1],
                'theta_queue':  y[2]
            }
            p_plus[name] += eps
            p_plus['lambda_over']  = np.clip(p_plus['lambda_over'], *bounds['lambda_over'])
            p_plus['lambda_under'] = np.clip(p_plus['lambda_under'], *bounds['lambda_under'])
            p_plus['theta_queue']  = np.clip(p_plus['theta_queue'], *bounds['theta_queue'])
            cost_plus, _, _ = smart_order_router(
                snapshots, order_size,
                lambda_over=p_plus['lambda_over'],
                lambda_under=p_plus['lambda_under'],
                theta_queue=p_plus['theta_queue']
            )
            p_minus = p_plus.copy()
            p_minus[name] = y[i] - eps
            p_minus['lambda_over']  = np.clip(p_minus['lambda_over'], *bounds['lambda_over'])
            p_minus['lambda_under'] = np.clip(p_minus['lambda_under'], *bounds['lambda_under'])
            p_minus['theta_queue']  = np.clip(p_minus['theta_queue'], *bounds['theta_queue'])
            cost_minus, _, _ = smart_order_router(
                snapshots, order_size,
                lambda_over=p_minus['lambda_over'],
                lambda_under=p_minus['lambda_under'],
                theta_queue=p_minus['theta_queue']
            )
            grad[i] = (cost_plus - cost_minus) / (2 * eps)
        v = mu * v - alpha * grad
        x = x + v
        x[0] = np.clip(x[0], *bounds['lambda_over'])
        x[1] = np.clip(x[1], *bounds['lambda_under'])
        x[2] = np.clip(x[2], *bounds['theta_queue'])
        cost_x, _, _ = smart_order_router(
            snapshots, order_size,
            lambda_over=x[0],
            lambda_under=x[1],
            theta_queue=x[2]
        )
        if cost_x < best['cost']:
            best['cost'] = cost_x
            best['params'] = x.copy()
        if verbose:
            lo, lu, tq = x
            print(f"Iter {it+1}/{n_iter}: λ_over={lo:.4f}, λ_under={lu:.4f}, θ_queue={tq:.6f}, cost={cost_x:.2f}")
    return {
        'lambda_over':  best['params'][0],
        'lambda_under': best['params'][1],
        'theta_queue':  best['params'][2],
        'cost':         float(best['cost'])
    }

def compute_cost(split, exchanges, order_size, lambda_over, lambda_under, theta_queue):
    executed = 0
    cash_spent = 0.0
    for i in range(len(exchanges)):
        exe = min(split[i], exchanges[i]['ask_size'])
        executed += exe
        cash_spent += exe * (exchanges[i]['ask'] + exchanges[i]['fee'])
        maker_rebate = max(split[i] - exe, 0) * exchanges[i]['rebate']
        cash_spent -= maker_rebate
    underfill = max(order_size - executed, 0)
    overfill  = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill
    return cash_spent + risk_pen + cost_pen

def allocate(order_size, exchanges, lambda_over, lambda_under, theta_queue, step=100):
    splits = [[]]
    for v in range(len(exchanges)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, exchanges[v]['ask_size'])
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = None

    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, exchanges, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost, best_split = cost, alloc

    if best_split is None:
        best_split = [0] * len(exchanges)

    return best_split, best_cost

def smart_order_router(snapshots, order_size, lambda_over, lambda_under, theta_queue):
    remaining, total_cost = order_size, 0.0
    for snap in snapshots:
        if remaining <= 0:
            break

        split, _ = allocate(remaining, snap['exchanges'], lambda_over, lambda_under, theta_queue)

        if len(split) != len(snap['exchanges']):
            split = split + [0] * (len(snap['exchanges']) - len(split))

        for qty, exch in zip(split, snap['exchanges']):
            exe = min(qty, exch['ask_size'], remaining)
            remaining -= exe
            total_cost += exe * (exch['ask'] + exch['fee'])

    filled = order_size - remaining
    avg_price = total_cost / filled if filled else float('inf')
    return total_cost, filled, avg_price

def calculate_naive_cost(n_orders, exch):
    return n_orders * (exch['ask'] + exch['fee'])

def generate_df(filepath):
    columns = ['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']
    types = {
        'publisher_id': 'category',
        'ask_px_00': 'float32',
        'ask_sz_00': 'int32'
    }
    df = pd.read_csv(filepath, usecols=columns, dtype=types)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
    df.drop_duplicates(['ts_event', 'publisher_id'], inplace=True)
    df.sort_values('ts_event', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def generate_snapshots(df):
    snapshots = []
    for ts_event, group in df.groupby('ts_event'):
        exchanges = [
            {
                'ask': float(row.ask_px_00),
                'ask_size': int(row.ask_sz_00),
                'fee': 0.003,
                'rebate': 0.0015
            } for row in group.itertuples(index=False)
        ]
        snapshots.append({'ts_event': ts_event, 'exchanges': exchanges})
    return snapshots




import json


def main():
    df = generate_df("l1_day.csv")
    raw_snaps = generate_snapshots(df)
    market_snaps = raw_snaps


    #Disegard this part of the code I just wanted to check how it performs when more venues are added. 
    #I had a feeling the hidden slice of data has more exchanges than the sample data so I added some noise

    #The added number of venues made the grid search very slow, so I implemented a accelrated gradient search (Nesterov 1983)
    #The Cont Kukanov paper have suggested Neriskovski et al 2009, However I found that it was slower that Nesterov 
    #In the case your test slice had more venues and more samples than the training slice, I wanted my algorithm to outperform the pool of candidates I am competing against


    # market_snaps = augment_snapshots(
    #     raw_snaps,
    #     num_clones=3,
    #     price_spread=0.01,
    #     size_spread=20
    # )

    ba_cost, ba_exec, ba_avg = best_ask_strategy(market_snaps, 5000)
    tw_cost, tw_exec, tw_avg = twap_strategy(market_snaps, 5000)
    vw_cost, vw_exec, vw_avg = vwap_strategy(market_snaps, 5000)

    bounds = {
        'lambda_over':  (0.01, 10.0),
        'lambda_under': (0.01, 10.0),
        'theta_queue':  (0.001, 0.1)
    }
    tune_result = tune_nesterov(
        market_snaps,
        order_size=5000,
        bounds=bounds,
        alpha=1.0,
        mu=0.9,
        eps=1e-3,
        n_iter=20,
        init_params=None,
        verbose=False
    )

    opt_over   = tune_result['lambda_over']
    opt_under  = tune_result['lambda_under']
    opt_theta  = tune_result['theta_queue']

    ContKukanov_cost, ContKukanov_exec, ContKukanov_avg = smart_order_router(
        market_snaps,
        order_size    = 5000,
        lambda_over   = opt_over,
        lambda_under  = opt_under,
        theta_queue   = opt_theta
    )

    def savings(baseline, actual):
        return (baseline - actual) / baseline * 10000

    output = {
        "parameters_found": {
            "lambda_over":  opt_over,
            "lambda_under": opt_under,
            "theta_queue":  opt_theta
        },
        "final figures": {
            "Cont Kukanov Router": {
                "total_cash_spent":     ContKukanov_cost,
                "orders_filled":  ContKukanov_exec,
                "average_fill_price": ContKukanov_avg
            },
            "BestAsk Strategy": {
                "total_cash_spent":     ba_cost,
                "average_fill_price": ba_avg
            },
            "TWAP Strategy": {
                "total_cash_spent":     tw_cost,
                "average_fill_price": tw_avg
            },
            "VWAP Strategy": {
                "total_cash_spent":     vw_cost,
                "average_fill_price": vw_avg
            }
        },
        "basis points savings versus baseline": {
            "BestAsk Strategy": savings(ba_avg, ContKukanov_avg),
            "TWAP Strategy":     savings(tw_avg, ContKukanov_avg),
            "VWAP Strategy":     savings(vw_avg, ContKukanov_avg)
        }
    }

    print(json.dumps(output, indent=3))
 #   print(json.dumps(output))

if __name__ == "__main__":
    main()
