# Known Issues

## Current Limitations

### 1. Azure PostgreSQL connectivity depends on external network rules
If the Azure database server is not configured for the client IP or the hosted app network path, the connection can fail even when the code and credentials are correct.

### 2. Dashboard maintainability
The Streamlit dashboard is feature-rich but still concentrated in a single large file. It works, but readability and long-term maintenance would improve if the UI were split into smaller modules.

### 3. Local generated outputs are not committed
The repository intentionally does not include generated datasets. A new user must run the pipeline locally before seeing populated outputs.

### 4. Live assignment to Autotask is not enabled
The ticket assignment board currently supports dashboard-side simulation stored in PostgreSQL. It does not push assignment actions back to Autotask.

### 5. Public deployment still needs cloud-side validation
The repository is much closer to deployment-ready now, but Azure hosting still depends on correct environment variables, package installation, startup configuration, and networking.
