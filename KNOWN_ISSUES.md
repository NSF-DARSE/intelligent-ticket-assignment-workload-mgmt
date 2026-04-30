# Known Issues

## Current Limitations

### 1. Azure PostgreSQL connectivity depends on external network rules
If the Azure database server is not configured for the client IP, the connection can time out even when the code is correct.

### 2. Dashboard complexity
The Streamlit dashboard is feature-rich but currently concentrated in a single large file. It works, but maintainability would improve if the UI is split into smaller modules.

### 3. Time estimation model quality
Time estimation is included as a workload-support signal, but it is not yet a highly accurate predictive model and should not be treated as a precise operational promise.

### 4. Local generated outputs are not committed
The repository intentionally does not include generated datasets. A new user must run the pipeline locally before seeing populated outputs.

### 5. Live assignment to Autotask is not enabled
The ticket assignment board currently supports dashboard simulation stored in PostgreSQL. It does not push assignment actions back to Autotask.

### 6. Public deployment still needs final cloud configuration
The repository is closer to deployment-ready now, but Azure hosting still requires cloud-side configuration for networking, environment variables, and startup behavior.
