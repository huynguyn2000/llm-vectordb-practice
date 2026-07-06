from core.models import Log

SAMPLE_LOGS = [
    # Normal DB logs
    Log(message="Database connection established successfully", level="INFO", service="user-service"),
    Log(message="Database connection pool initialized with 10 connections", level="INFO", service="order-service"),
    Log(message="Database query executed in 12ms", level="DEBUG", service="user-service"),
    Log(message="Database connection refused: timeout after 30s", level="ERROR", service="payment-service"),
    Log(message="Lost connection to database server during query", level="ERROR", service="inventory-service"),
    Log(message="Database connection pool exhausted, waiting for available connection", level="WARN", service="user-service"),

    # Normal HTTP logs
    Log(message="GET /api/users returned 200 OK in 45ms", level="INFO", service="api-gateway"),
    Log(message="POST /api/orders returned 201 Created in 120ms", level="INFO", service="api-gateway"),
    Log(message="GET /api/products returned 200 OK in 30ms", level="INFO", service="api-gateway"),
    Log(message="PUT /api/users/123 returned 400 Bad Request", level="WARN", service="api-gateway"),
    Log(message="DELETE /api/sessions returned 401 Unauthorized", level="WARN", service="api-gateway"),

    # Normal service logs
    Log(message="Cache hit for key user:profile:42", level="DEBUG", service="cache-service"),
    Log(message="Cache miss for key product:detail:99, fetching from database", level="DEBUG", service="cache-service"),
    Log(message="Email notification sent to user@example.com", level="INFO", service="notification-service"),
    Log(message="Scheduled job 'cleanup-expired-sessions' completed in 2.3s", level="INFO", service="scheduler"),

    # Anomalous / unusual logs
    Log(message="Memory usage exceeded 95% threshold: heap dump initiated", level="ERROR", service="analytics-service"),
    Log(message="Unexpected binary data received on port 8472, connection dropped", level="ERROR", service="api-gateway"),
    Log(message="SSH login attempt from IP 192.168.99.254 with invalid credentials (attempt 47)", level="WARN", service="auth-service"),
    Log(message="Kernel panic: unable to mount root fs on unknown-block(0,0)", level="ERROR", service="host-agent"),
    Log(message="Circuit breaker OPEN for downstream service payment-gateway after 5 consecutive failures", level="ERROR", service="payment-service"),
]
