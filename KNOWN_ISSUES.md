# Known Issues

## Current Limitations

### 1. Generated datasets are local-only

The repository does not commit generated datasets under `data/`. A fresh user must run the pipeline before the dashboard has populated reporting tables.

### 2. MiniLM may need a first-run download

The full NLP similarity step requires `sentence-transformers/all-MiniLM-L6-v2`. The benchmark can reuse cached local similarity outputs, but a full recomputation needs the model downloaded or already cached.

### 3. Dashboard maintainability

The Streamlit dashboard is feature-rich but still concentrated in one large file. It is functional, but future maintainability would improve by splitting it into smaller view/data/helper modules.

### 4. Dispatch actions are simulated

Dashboard assignment actions are stored in local PostgreSQL table `autotask_dashboard_dispatch_actions`. They do not update Autotask itself.

### 5. Local PostgreSQL is required

The active project intentionally targets local PostgreSQL only. `DB_HOST` must be `localhost`, `127.0.0.1`, or `::1`.
