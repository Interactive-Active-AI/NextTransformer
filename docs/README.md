# NextTransformer documentation

The documentation site, built with [Starlight](https://starlight.astro.build/). It is deployed together with the
observatory to GitHub Pages, under `/docs/` (see `.github/workflows/pages.yml`).

```bash
cd docs
npm ci
npm run dev       # http://localhost:4321/NextTransformer/docs/
npm run build     # static site in docs/dist; DOCS_BASE=/<repository>/docs/ for a fork
```

- English pages live in `src/content/docs/`, Chinese pages at the same paths under `src/content/docs/zh-cn/`. Keep an English and a Chinese source for each page. CI checks that each English route has a Chinese counterpart.
- The protocol pages are normative in English; changing them changes the training contract.
- `methods/zoo.md`, `methods/configurations.md` and the leaderboard tables are generated from `results/` and `configs/`
  by `python docs/scripts/generate.py`; CI checks that they are up to date.
- Use relative links between pages, so the site works under any base path.
