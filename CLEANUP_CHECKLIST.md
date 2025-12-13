# ✅ CLEANUP CHECKLIST — Complete

## Pre-Cleanup

- [ ] Read CLEANUP_EXECUTION_GUIDE.md (this guide)
- [ ] Backup repository (optional but recommended)
- [ ] Activate virtual environment
- [ ] Have pytest ready

---

## The 17 Files to Delete

### Legacy Dashboards (1)
- [ ] dashboard.py

### Duplicates (2)
- [ ] models_mod.py
- [ ] payoffs_mod.py

### Advanced/Incomplete (4)
- [ ] calibration.py
- [ ] model_risk.py
- [ ] hedge.py
- [ ] compute_backend.py

### Test Runners (8)
- [ ] run_greeks_smoke.py
- [ ] run_local_tests.py
- [ ] run_pytest_programmatic.py
- [ ] run_single_greeks_test.py
- [ ] run_tests_explicit.py
- [ ] smoke_test.py
- [ ] smoke_test_runner.py
- [ ] quickstart.py

**Total: 17 files**

---

## Execution Checklist

### Step 1: Choose Deletion Method
- [ ] Option A (PowerShell - Fastest)
- [ ] Option B (Git - Safest)
- [ ] Option C (Manual - Careful)

### Step 2: Execute Deletion
```powershell
# Copy from CLEANUP_EXECUTION_GUIDE.md
# Paste into PowerShell
# Run command
```
- [ ] Command copied
- [ ] PowerShell ready
- [ ] Delete command executed

### Step 3: Verify Tests Pass
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
- [ ] All tests PASS ✅
- [ ] No failures
- [ ] No errors

### Step 4: Verify Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard_v2.py
```
- [ ] Dashboard opens
- [ ] No import errors
- [ ] UI responsive

### Step 5: Final Check
```powershell
(Get-ChildItem -Recurse -File).Count
# Should show ~93
```
- [ ] File count reduced
- [ ] Documentation preserved
- [ ] Config files intact

---

## Post-Cleanup

- [ ] All tests pass
- [ ] Dashboard works
- [ ] Documentation intact
- [ ] Repository cleaned ✅

---

## Rollback Plan (If Needed)

If something goes wrong:

```powershell
# If using Option B (Git), restore files:
git checkout HEAD~1

# If using Option A/C, restore from backup or git history
git checkout HEAD -- <file>
```

---

## Success Criteria

After cleanup, verify:

✅ **Tests**: All pass (`pytest tests/ -v`)
✅ **Dashboard**: Opens without errors (`streamlit run dashboard_v2.py`)
✅ **Imports**: No broken imports
✅ **Files**: 17 files removed, ~93 remaining
✅ **Functionality**: Zero behavior changes

---

## Summary

**Files to delete**: 17
**Time to execute**: 5 minutes
**Risk level**: Very Low (documented, verified, reversible)
**Breaking changes**: None

**All documentation provided in root directory.**

---

**✅ READY TO CLEAN UP!**

Copy deletion command from CLEANUP_EXECUTION_GUIDE.md and execute.


