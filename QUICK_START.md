# 🚀 Quick Start — Systematix Pro Dashboard

## Step 1: Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your prompt.

## Step 2: Run the Dashboard
```powershell
# Start Streamlit app
.\.venv\Scripts\python.exe -m streamlit run dashboard_v2.py
```

The app will open automatically in your browser at `http://localhost:8501`

If not, copy/paste the URL shown in the terminal.

## Step 3: Explore Models

### 3a. GBM (Fastest & Simplest)
- **Model**: Select "GBM"
- **Strike**: K = 100
- **Volatility**: σ = 0.20
- **Time**: T = 1.0 year
- **Paths**: 2000 (default)
- **Click**: "RUN SIMULATION"
- **See**: Price in Middle column
- **Try**: Click "Compute Greeks" → View Delta, Gamma, Vega

### 3b. Heston (Stochastic Vol)
- **Model**: Select "Heston"
- **Extra Params**: v0 = 0.04, κ = 1.5, θ = 0.04, ξ = 0.3, ρ = -0.7
- **Click**: "RUN SIMULATION"
- **Observe**: Price differs from GBM (vol clustering effect)
- **Try**: Vary ρ from 0.0 to -0.9 → See how correlation affects price

### 3c. Merton (Jump Risk)
- **Model**: Select "Merton Jump"
- **Extra Params**: λ = 0.1 (jumps/year), μ_J = 0.0, σ_J = 0.3
- **Click**: "RUN SIMULATION"
- **Observe**: Higher price than GBM (tail risk premium)
- **Try**: Vary λ → See how jump intensity affects price

### 3d. Kou (Asymmetric Jumps)
- **Model**: Select "Kou Jump"
- **Extra Params**: λ = 0.1, p = 0.5, η⁺ = 1.0, η⁻ = 2.0
- **Click**: "RUN SIMULATION"
- **Observe**: More extreme pricing due to exponential jump sizes

### 3e. G2++ (Rate Model)
- **Model**: Select "G2++"
- **Extra Params**: r0 = 0.03, a = 0.1, σ = 0.015, η = 0.025, ρ = 0.8
- **Click**: "RUN SIMULATION"
- **Note**: This simulates interest rates (not equity prices)

## Step 4: Custom Payoffs

### Example 1: Straddle (Long Call + Long Put)
```python
def custom_payoff(S):
    K = 100
    return np.maximum(S - K, 0) + np.maximum(K - S, 0)
```

**Steps**:
1. Check "Use custom payoff" in left column
2. Replace default code with above
3. Click "RUN SIMULATION"
4. See history update with "Custom" option type

### Example 2: Digital Option
```python
def custom_payoff(S):
    return np.where(S > 100, 1.0, 0.0)
```

### Example 3: Barrier (Up-and-Out Call)
```python
def custom_payoff(S):
    # S shape: (n_paths, steps+1)
    # Terminal payoff
    payoff = np.maximum(S[:, -1] - 100, 0)
    # Knock out if ever hit 120
    hit_barrier = np.any(S > 120, axis=1)
    payoff[hit_barrier] = 0
    return payoff
```

## Step 5: Compute Greeks

1. After running a simulation, scroll down in Middle column
2. Click "Compute Greeks" button
3. See:
   - **Δ (Delta)**: Price sensitivity to spot (∂Price/∂S)
   - **Γ (Gamma)**: Delta sensitivity (∂²Price/∂S²)
   - **ν (Vega)**: Price sensitivity to vol (∂Price/∂σ)
   - **ρ (Rho)**: Price sensitivity to rates (∂Price/∂r)
   - **Θ (Theta)**: Price sensitivity to time (∂Price/∂T)

**Interpretation**:
- Δ > 0 = bullish (call is ITM)
- Δ ≈ 0.5 = at-the-money
- Γ > 0 = convex (good for volatile markets)
- ν > 0 = long volatility
- Θ < 0 = time decay (theta decay on long options)

## Step 6: Track History

- Right column shows all past simulations
- **Download**: Click "Download history (CSV)" to export for Excel/analysis
- **Clear**: Click "🗑️ Clear History" to start fresh

## Examples Workflow

### Example A: Compare GBM vs Heston
1. Run GBM: S0=100, K=100, σ=0.20 → Note price
2. Run Heston: Same params + v0=0.04, κ=1.5, θ=0.04, ξ=0.3, ρ=-0.7 → Note price
3. Check History: See price difference (usually Heston < GBM for ATM calls)

### Example B: Vol Smile (Heston)
1. Run Heston Call: K=90 (ITM) → Note price
2. Run Heston Call: K=100 (ATM) → Note price
3. Run Heston Call: K=110 (OTM) → Note price
4. Decrease ρ (make more negative) → Rerun all 3 → See smile steepen

### Example C: Greeks Sensitivity
1. Run GBM with 2000 paths → Compute Greeks
2. Rerun with 5000 paths → Compute Greeks again
3. Compare stderr values: Higher paths → Lower error on Greeks

### Example D: Custom Barrier Option Hedging
1. Define barrier payoff in custom code
2. Price the option (left 1/4)
3. View terminal distribution (middle)
4. Compute Delta (middle Greeks section)
5. Use Delta to set up hedge ratio

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Rerun script (after code changes) |
| `C` | Clear cache (if results look stale) |
| `Ctrl+C` (terminal) | Stop Streamlit server |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App very slow | ↓ n_paths (try 1000), ↓ steps (try 6) |
| Custom payoff error | Use simple payoff first: `np.maximum(S - 100, 0)` |
| Greeks have huge stderr | ↑ n_paths to 5000, ensure payoff smooth |
| G2++ doesn't show equity prices | G2++ is for rates; use GBM/Heston/Merton for equities |
| App won't start | Check: `.venv\Scripts\Activate.ps1` ran, `pip install streamlit` worked |

## Next Steps

1. **Read** `DASHBOARD_GUIDE.md` for detailed model documentation
2. **Experiment** with parameters in the dashboard
3. **Export** history for analysis in Excel/Python
4. **Explore** custom payoff functions
5. **Compare** models side-by-side

---

## Advanced: Run Unit Tests

```powershell
# Test all models
.\.venv\Scripts\python.exe test_all_models.py

# Expected output:
# 1. GBM: Price: X.XXXXXX, SE: X.XXXXXX
# 2. Heston: Price: X.XXXXXX, SE: X.XXXXXX
# 3. Merton: Price: X.XXXXXX, SE: X.XXXXXX
# 4. Kou: Price: X.XXXXXX, SE: X.XXXXXX
# 5. Greeks: Delta: X, Gamma: X, Vega: X
# ✅ All models working!
```

---

**Questions?** Check `DASHBOARD_GUIDE.md` or `IMPLEMENTATION_SUMMARY.md` for deeper docs.

**Happy pricing! 📈**

