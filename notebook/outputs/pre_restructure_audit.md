# Pre-Restructure Audit

## Actual roots
- Root: `w:\3 projects\Building\Tfrenzy`
- Working App Root: `notebook`
- Frontend Root: `notebook/frontend`
- Backend Root: `notebook/backend`
- ML Pipeline Root: `notebook/src`
- Model Dir: `notebook/models`
- Dataset Dir: `notebook/datasets`
- Deployment Dir: `notebook/deployment`
- Documentation Dir: `docs` and `notebook/outputs`

## Hardcoded `W:\` Paths Found
- `notebook\runs\detect\runs\ppe_extension\v3_boots\args.yaml:116: save_dir: W:\3 projects\Building\Tfrenzy\notebook\runs\detect\runs\ppe_extension\v3_boots`
- No hardcoded absolute Windows paths were found in the core `src`, `backend`, or `frontend` code that would break the pipeline upon relocation.
