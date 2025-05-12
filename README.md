# Multiple-Venue-Optimal-Order-Placement Backtest
Smart Order Router Backtest

## Files

- **backtest.py**  
  - **Dependencies**: only `numpy`, `pandas`, and Python standard library  
  - **Workflow**:  
    1. **Load & Clean** `l1_day.csv` → drop duplicates, sort by timestamp  
    2. **Snapshot Generation** → group by `ts_event`, build list of `Venue(ask, size, fee, rebate)`  
    3. **Baselines**  
       - **Best-Ask**: always hit the lowest ask, share-by-share  
       - **TWAP**: 60 s buckets, allocate evenly, fill via Best-Ask within each bucket  
       - **VWAP**: proportional to displayed depth, share-by-share  
    4. **Smart Order Routing Implementation**  
       - Cont & Kukanov static allocator per snapshot  
       - Cost = cash + queue‐risk + over/underfill penalties  
    5. **Parameter Tuning**  
       - **Nesterov’s accelerated gradient** (faster than grid)  
       - Tunes λ_over, λ_under, θ_queue to minimize total cost  
    6. **Results & JSON**  
       - Prints best parameters, total cost & avg fill price for SOR and baselines  
       - Separate “basis points savings against baseline” section  

- **README.md** (this file)  


## Search Choices

- THe grid and random searches worked well for the current dataset, however I augmented the dataset with more noise and added simulated exchanges to test for robustness. The searches exceeded the two minute limit. Thus I had to improvise. I tried Neverowski method as mentioned in the paper however settled at Nersterov's 1983 accerated gradient descent
- **Nesterov’s Method** gave rapid convergence with minimal backtests.  Switches from O(n^3) to much lesser search space. I want to say linear but can't confirm.


## Suggested Improvement

I would say using the Nesterov hyper parameter tuning itself was an improvement however 
While the static Cont & Kukanov model + Nesterov tuning performs well on historical data, making the hyper-parameter search **adaptive** would allow the router to react to changing market conditions in real time. For example:

- Use a lightweight **Bayesian Optimization** or **CMA-ES** process running in parallel to periodically re-tune (λ_over, λ_under, θ_queue) based on the last N minutes of fills.  
- This ensures the SOR remains calibrated to current volatility, liquidity fragmentation, and venue behavior—further reducing slippage and market impact.

