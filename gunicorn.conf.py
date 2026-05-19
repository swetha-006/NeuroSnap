import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Workers
workers = 2                          # 2 is safe for free tier (512MB RAM)
worker_class = "sync"
worker_connections = 1000
timeout = 120                        # 120s timeout (ML inference can be slow)
keepalive = 2

# Logging
errorlog = "-"                       # stderr
loglevel = "info"
accesslog = "-"                      # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s"'

# Process naming
proc_name = "neurosnap"

# Server mechanics
preload_app = True                   # load model ONCE before forking workers
