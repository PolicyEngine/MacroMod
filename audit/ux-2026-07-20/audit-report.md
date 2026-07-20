# PolicyEngine Macro PR #63 — navigation and information architecture audit

## Audit scope

Combined UX and screenshot-based accessibility review of the open PR, focused on the Home, Compare, Validation, Papers, and model-detail routes at desktop, plus Compare at 390px.

User goal: quickly answer three questions without losing context:

1. Which model should I use?
2. What evidence or paper supports it?
3. How well has it been validated?

## Overall verdict

The PR improves model discovery, but it does not yet give the content a stable information architecture. Models, comparison, papers, and validation are presented as related concepts, but live at inconsistent levels: Models is a homepage anchor, Compare and Validation are top-level routes, Papers is a hidden/legacy route, and model evidence is split between model pages, Compare, Papers, and Validation.

The clearest fix is a persistent secondary navigation on every research-facing page:

`Models | Compare | Papers | Validation`

Keep the global header for product-level destinations (`Explore`, `Get started`, GitHub). On desktop and mobile, the secondary navigation should remain visible, show the current location, and preserve the same order everywhere.

## Flow review

### Step 1 — Home / discover a model — Health: mixed

![Home page](01-home.png)

Strengths: question-first model discovery and the five-card catalog are clear. The page also introduces validation early enough to build trust.

Risks: “Models” is an anchor on Home while Compare and Validation are full routes. Papers is absent from the header. That makes four sibling concepts look like three different kinds of destination. The homepage also duplicates substantial validation content, weakening the purpose of the dedicated Validation page.

Recommendation: make `/models/` or the model catalog a first-class destination; keep only a short validation summary on Home.

### Step 2 — Compare models — Health: needs restructuring

![Compare page](02-compare.png)

Strengths: the question-first chooser and side-by-side model matrix support selection well.

Risks: the page title asks a model-selection question, the eyebrow says “Documentation,” and the header says “Compare.” These three labels compete. “References and papers” is placed at the bottom of Compare, so Papers behaves like a section, even though `/papers/` still exists. Validation is only an inline link and feels detached from each model.

Recommendation: label the route consistently as Compare; add the persistent secondary navigation; keep model cards or model links directly below the matrix; move the paper library to Papers and link each comparison column to its paper and validation record.

### Step 3 — Validation — Health: valuable evidence, difficult to navigate

![Validation page](03-validation.png)

Strengths: the evidence is unusually transparent, model-specific, and honest about different validation classes.

Risks: the page is extremely long and has no local navigation, selected model, sticky contents, or progressive disclosure. Users must scroll through every model to reach the one they care about. The page alternates overview, charts, tables, and caveats without a stable repeated template.

Recommendation: add `Overview | OBR | BoE | FRB/US | PolicyEngine | PSL OLG` as a model-level tab/filter below the persistent secondary navigation. Give every model the same content order: status, benchmark, headline result, chart, limitations, paper/source links.

### Step 4 — Papers — Health: broken information scent

![Papers page](04-papers.png)

Strengths: the copy explains that papers are connected to models.

Risks: a visible route whose headline says “The papers moved into the docs” feels like a redirect page that was never retired. “Docs,” “documentation,” “model library,” “reference,” and “working paper” are used as overlapping names. Users cannot confidently predict whether a link opens a model guide, paper landing page, or PDF.

Recommendation: restore Papers as a real sibling destination, not a relocation notice. Show one card per paper with model, institution, country, paper title, publication date, validation type, and clear actions: `Read summary`, `View PDF`, `View model`.

### Step 5 — Model detail — Health: strong content, weak orientation

![Model detail](05-model.png)

Strengths: the model page contains the method, workflow, validation summary, and limitations needed for a serious decision.

Risks: once inside a model, users lose the relationship to the model catalog, comparison, paper, and full validation record. The long page lacks an in-page contents control. Evidence exists, but the paths between evidence surfaces are implicit.

Recommendation: add a breadcrumb (`Models / OBR macro`) and local model navigation: `Overview | Method | Run | Validation | Paper`. Also keep cross-library links to Compare and all Models.

### Step 6 — Compare on mobile — Health: poor

![Compare page at 390px](06-compare-mobile.png)

Strengths: the question chooser and model cards reflow into readable single-column blocks.

Risks: Models, Compare, and Validation disappear from the mobile header, leaving only Get started and GitHub. The comparison table is clipped into a narrow horizontal slice, with no visible cue that more columns exist. Small all-caps labels and dense table text are likely difficult at zoom.

Recommendation: expose the persistent secondary navigation as a horizontally scrollable tab row with a clear active state. Replace the mobile matrix with a two-model comparison control or stacked attribute cards; do not rely on an unlabelled horizontal table scroll.

## Recommended site structure

```text
Home
Explore
├── Models
│   ├── All models
│   └── Model detail
├── Compare
├── Papers
└── Validation
Get started
├── Hosted tools
├── CLI
└── Python
```

Persistent Explore subnav on Models, Compare, Papers, and Validation:

`Models | Compare | Papers | Validation`

Model-detail local navigation:

`Overview | Method | Run | Validation | Paper`

Validation model filter:

`Overview | OBR | BoE | FRB/US | PolicyEngine | PSL OLG`

## Highest-impact changes

1. Add one persistent Explore subnav across Models, Compare, Papers, and Validation, including mobile and `aria-current="page"`.
2. Make Papers a real destination and remove the “papers moved” holding page.
3. Break Validation into model-filtered sections with a consistent evidence template.
4. Add breadcrumbs and local section navigation to every model detail page.
5. Replace the mobile comparison matrix with a deliberate mobile comparison pattern.
6. Standardize naming: use Models, Compare, Papers, Validation, and Get started everywhere; reserve “docs” for technical/reference documentation only.

## Accessibility risks and verification limits

Visible risks include missing current-location cues, hidden primary destinations on mobile, dense small table text, and a clipped comparison table without a visible interaction cue. Screenshots cannot confirm keyboard access, focus visibility, semantic tab behavior, screen-reader labels, or contrast ratios. Those require DOM and keyboard testing after the navigation structure is implemented.

## Evidence limits

The Vercel preview required authentication, so the accepted screenshots were captured from the exact PR branch served locally. This review covers the named routes and one mobile viewport; it is not a full WCAG conformance audit.
