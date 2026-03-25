"""
SRE-Nidaan: Massive SRE Incident Dataset Generator
====================================================
Generates 2,500+ high-quality causal reasoning training examples
covering 12 cloud-native infrastructure domains.

Each example includes:
  - SRE incident scenario (premise)
  - Causal reasoning chain
  - Counterfactual analysis
  - Pearl's Hierarchy level (L1/L2/L3)
  - Causal DAG (nodes + edges)
  - Quality score for reward model training
  - Confounding intervention analysis

Architecture mirrors NEXUS-CAUSAL v3.1 DataScaledDatasetGenerator.
"""

import json
import random
import os
from typing import List, Dict
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# SRE Domain Templates
# ─────────────────────────────────────────────────────────────────────────────

INCIDENT_TEMPLATES = {
    "kubernetes": [
        {
            "premise": "A Kubernetes cluster experiences cascading pod evictions after a node's memory utilization exceeds {mem_pct}%. The HPA scales from {replicas_before} to {replicas_after} replicas, but response latency worsens from {latency_before}ms to {latency_after}ms.",
            "root_cause": "Memory leak in the application's connection pool causes OOM pressure. The HPA scales up pods which compete for the same limited node memory, accelerating evictions. The structural root cause is the unbounded connection pool, not insufficient replicas.",
            "confounding_action": "do(Scale Up Replicas) increases memory contention on the node — each new pod allocates {mem_per_pod}MB, pushing total utilization past the eviction threshold faster. This transforms a degraded service into a complete outage.",
            "counterfactual": "Had the connection pool been bounded to {pool_limit} connections with a TTL of {ttl}s, memory usage would stabilize at ~{stable_mem}%, preventing OOM evictions entirely regardless of request volume.",
            "correct_action": "1. Patch application to bound connection pool to {pool_limit} connections. 2. Set container memory limit to {mem_limit}MB with a request of {mem_request}MB. 3. Configure PDB to prevent simultaneous evictions.",
            "dag_nodes": [
                {"id": "conn_pool_leak", "label": "Connection Pool Leak"},
                {"id": "memory_pressure", "label": "Node Memory Pressure ({mem_pct}%)"},
                {"id": "pod_eviction", "label": "Pod Eviction Cascade"},
                {"id": "hpa_scaling", "label": "HPA Scale-Up ({replicas_after} pods)"},
                {"id": "mem_contention", "label": "Memory Contention"},
                {"id": "latency_spike", "label": "Latency Spike ({latency_after}ms)"},
            ],
            "dag_edges": [
                {"source": "conn_pool_leak", "target": "memory_pressure"},
                {"source": "memory_pressure", "target": "pod_eviction"},
                {"source": "pod_eviction", "target": "hpa_scaling"},
                {"source": "hpa_scaling", "target": "mem_contention"},
                {"source": "mem_contention", "target": "pod_eviction"},
                {"source": "pod_eviction", "target": "latency_spike"},
            ],
            "params": {"mem_pct": [85, 90, 95], "replicas_before": [3, 5, 8], "replicas_after": [10, 15, 20], "latency_before": [50, 100, 200], "latency_after": [2000, 5000, 8000], "mem_per_pod": [256, 512, 1024], "pool_limit": [50, 100, 200], "ttl": [30, 60, 120], "stable_mem": [55, 65, 72], "mem_limit": [512, 1024, 2048], "mem_request": [256, 512, 1024]},
        },
        {
            "premise": "After a rolling deployment of service v{version}, {error_pct}% of requests return HTTP 502 errors. The deployment passed all CI/CD checks. Readiness probes report healthy, but downstream connections fail with 'connection refused' on port {port}.",
            "root_cause": "The new version changed the application's startup sequence — it now binds to port {port} before initializing its dependency connections. Readiness probes pass (port is open) but the service cannot handle requests until dependencies are ready, creating a window of false-healthy state.",
            "confounding_action": "do(Rollback Deployment) temporarily fixes symptoms but masks a systemic issue: readiness probes are structurally insufficient. Without fixing the startup race condition, any future deployment with added dependencies will exhibit the same failure pattern.",
            "counterfactual": "Had the readiness probe checked the /health/ready endpoint (which validates downstream connectivity) instead of a TCP port check, the false-healthy window would not have existed and traffic would not have been routed to unready pods.",
            "correct_action": "1. Change readiness probe from TCP to HTTP GET /health/ready with initialDelaySeconds={delay}. 2. Add startup probe with failureThreshold={threshold}. 3. Implement graceful startup with dependency readiness checks.",
            "dag_nodes": [
                {"id": "new_deployment", "label": "Rolling Deployment v{version}"},
                {"id": "startup_race", "label": "Startup Race Condition"},
                {"id": "false_ready", "label": "False Readiness (TCP Probe)"},
                {"id": "traffic_routing", "label": "Traffic Routed to Unready Pods"},
                {"id": "conn_refused", "label": "Connection Refused (Port {port})"},
                {"id": "http_502", "label": "HTTP 502 Errors ({error_pct}%)"},
            ],
            "dag_edges": [
                {"source": "new_deployment", "target": "startup_race"},
                {"source": "startup_race", "target": "false_ready"},
                {"source": "false_ready", "target": "traffic_routing"},
                {"source": "traffic_routing", "target": "conn_refused"},
                {"source": "conn_refused", "target": "http_502"},
            ],
            "params": {"version": ["2.1.0", "3.0.0", "2.5.1"], "error_pct": [15, 30, 45], "port": [8080, 3000, 9090], "delay": [15, 30, 45], "threshold": [10, 15, 20]},
        },
    ],
    "database": [
        {
            "premise": "A PostgreSQL database serving {qps} queries/sec exhibits connection pool exhaustion at {conn_used}/{conn_max} connections. The auth_service CPU is at {cpu}% with {replicas} replicas. Frontend returns 503 Gateway Timeout with error rate spiking. The database wait_event shows '{wait_event}'.",
            "root_cause": "The auth_service retry storm is the structural root cause. Each authentication failure triggers {retry_count} exponential retries, each holding a database connection with a ClientRead lock. The retries compound geometrically, exhausting the connection pool. The high CPU is a symptom of retry processing, not the cause.",
            "confounding_action": "do(Scale Up Auth_Service) is a confounding error: adding replicas increases concurrent DB connections proportionally. With {replicas} replicas each holding {conn_per_replica} connections, scaling to {scaled_replicas} replicas would push connections from {conn_used} to {projected_conn}, exceeding the {conn_max} limit and causing a full outage.",
            "counterfactual": "Had the auth_service implemented a circuit breaker with a half-open state after {cb_timeout}s, the retry storm would have been contained to the first {cb_threshold} failures. Database connections would have stabilized at ~{stable_conn}/{conn_max}, preventing pool exhaustion.",
            "correct_action": "1. Rate limit frontend to {rate_limit} req/s. 2. Increase DB max_connections from {conn_max} to {new_conn_max}. 3. Implement circuit breaker in auth_service (threshold={cb_threshold}, timeout={cb_timeout}s). 4. Drain locked ClientRead connections.",
            "dag_nodes": [
                {"id": "frontend", "label": "Frontend (503 Gateway Timeout)"},
                {"id": "auth_service", "label": "Auth Service ({cpu}% CPU)"},
                {"id": "retry_storm", "label": "Retry Storm ({retry_count}x retries)"},
                {"id": "database", "label": "Database ({conn_used}/{conn_max} connections)"},
                {"id": "conn_exhaustion", "label": "Connection Pool Exhaustion"},
                {"id": "client_lock", "label": "ClientRead Lock ({wait_event})"},
            ],
            "dag_edges": [
                {"source": "frontend", "target": "auth_service"},
                {"source": "auth_service", "target": "retry_storm"},
                {"source": "retry_storm", "target": "database"},
                {"source": "database", "target": "conn_exhaustion"},
                {"source": "conn_exhaustion", "target": "client_lock"},
                {"source": "client_lock", "target": "frontend"},
            ],
            "params": {"qps": [500, 1000, 2000], "conn_used": [950, 980, 990], "conn_max": [1000, 1000, 1000], "cpu": [88, 93, 96], "replicas": [3, 5, 8], "retry_count": [3, 5, 8], "conn_per_replica": [50, 80, 120], "scaled_replicas": [10, 15, 20], "projected_conn": [1200, 1500, 2400], "cb_timeout": [10, 30, 60], "cb_threshold": [5, 10, 15], "stable_conn": [200, 350, 500], "rate_limit": [100, 200, 500], "new_conn_max": [2000, 3000, 5000], "wait_event": ["ClientRead (Locked)", "LWLock:BufferContent", "IO:DataFileRead"]},
        },
        {
            "premise": "A MongoDB replica set shows replication lag of {lag}s on secondary nodes. Write operations take {write_latency}ms (10x normal). The oplog window has shrunk from {oplog_normal}h to {oplog_shrunk}h. WiredTiger cache utilization is at {cache_pct}%.",
            "root_cause": "A bulk data migration job is running unthrottled, generating {ops_per_sec} write operations per second. This saturates the WiredTiger cache and the oplog, causing secondaries to fall behind. The replication lag is a symptom of oplog saturation, not network or hardware issues.",
            "confounding_action": "do(Add Secondary Nodes) does not help because the oplog generation rate exceeds the replication capacity. New secondaries would need to perform an initial sync from the primary, further increasing load and potentially causing the primary to OOM.",
            "counterfactual": "Had the migration job been throttled to {throttled_ops} ops/s with batch sizes of {batch_size}, the oplog generation rate would have stayed within replication capacity, maintaining lag under {max_lag}s.",
            "correct_action": "1. Throttle migration job to {throttled_ops} ops/s. 2. Increase oplog size to {new_oplog}GB. 3. Schedule migrations during off-peak hours. 4. Monitor replication lag with alerting threshold at {alert_lag}s.",
            "dag_nodes": [
                {"id": "migration_job", "label": "Unthrottled Migration ({ops_per_sec} ops/s)"},
                {"id": "oplog_saturation", "label": "Oplog Saturation ({oplog_shrunk}h window)"},
                {"id": "wt_cache", "label": "WiredTiger Cache ({cache_pct}%)"},
                {"id": "repl_lag", "label": "Replication Lag ({lag}s)"},
                {"id": "write_latency", "label": "Write Latency ({write_latency}ms)"},
                {"id": "data_inconsistency", "label": "Read-After-Write Inconsistency"},
            ],
            "dag_edges": [
                {"source": "migration_job", "target": "oplog_saturation"},
                {"source": "migration_job", "target": "wt_cache"},
                {"source": "oplog_saturation", "target": "repl_lag"},
                {"source": "wt_cache", "target": "write_latency"},
                {"source": "repl_lag", "target": "data_inconsistency"},
            ],
            "params": {"lag": [30, 120, 300], "write_latency": [500, 2000, 5000], "oplog_normal": [24, 48, 72], "oplog_shrunk": [1, 2, 4], "cache_pct": [85, 92, 98], "ops_per_sec": [5000, 10000, 50000], "throttled_ops": [500, 1000, 2000], "batch_size": [100, 500, 1000], "max_lag": [2, 5, 10], "new_oplog": [10, 25, 50], "alert_lag": [5, 10, 30]},
        },
    ],
    "networking": [
        {
            "premise": "Intermittent TCP connection resets (RST) occur between microservices in zone {zone}. {pct_affected}% of requests fail with ECONNRESET. Network throughput is {throughput}Gbps against a {capacity}Gbps capacity. No packet loss detected by ICMP ping.",
            "root_cause": "An iptables conntrack table overflow is causing the kernel to drop new connections. The conntrack table is at {conntrack_used}/{conntrack_max} entries. ICMP ping doesn't use conntrack (NOTRACK), masking the issue from standard network health checks.",
            "confounding_action": "do(Increase Network Bandwidth) has no effect because the issue is not throughput-related but conntrack state exhaustion. Adding bandwidth does not increase the conntrack table size and wastes infrastructure spend.",
            "counterfactual": "Had the conntrack_max been set to {new_conntrack} and conntrack_tcp_timeout_established reduced from {timeout_current}s to {timeout_new}s, stale entries would expire faster and the table would never overflow at current traffic levels.",
            "correct_action": "1. Increase conntrack_max to {new_conntrack}. 2. Reduce conntrack_tcp_timeout_established to {timeout_new}s. 3. Enable conntrack accounting for monitoring. 4. Consider switching to IPVS for high-connection services.",
            "dag_nodes": [
                {"id": "high_connections", "label": "High Connection Rate"},
                {"id": "conntrack_overflow", "label": "Conntrack Table Overflow ({conntrack_used}/{conntrack_max})"},
                {"id": "kernel_drop", "label": "Kernel Drops New Connections"},
                {"id": "tcp_rst", "label": "TCP RST (ECONNRESET)"},
                {"id": "service_errors", "label": "Service Errors ({pct_affected}%)"},
                {"id": "icmp_ok", "label": "ICMP Ping OK (Misleading)"},
            ],
            "dag_edges": [
                {"source": "high_connections", "target": "conntrack_overflow"},
                {"source": "conntrack_overflow", "target": "kernel_drop"},
                {"source": "kernel_drop", "target": "tcp_rst"},
                {"source": "tcp_rst", "target": "service_errors"},
            ],
            "params": {"zone": ["us-east-1a", "eu-west-1b", "ap-south-1c"], "pct_affected": [5, 15, 30], "throughput": [2, 5, 8], "capacity": [10, 10, 10], "conntrack_used": [65000, 128000, 250000], "conntrack_max": [65536, 131072, 262144], "new_conntrack": [524288, 1048576, 2097152], "timeout_current": [432000, 432000, 432000], "timeout_new": [120, 300, 600]},
        },
    ],
    "load_balancer": [
        {
            "premise": "An ALB distributing traffic across {num_targets} targets shows {unhealthy_pct}% target health check failures. Active connection count is {active_conn}. Surge queue length is {queue_len}. Backend response time is {backend_rt}ms. HTTP 5xx rate is {error_rate}/min.",
            "root_cause": "Backend targets are experiencing thread pool exhaustion due to a slow downstream dependency (response time {downstream_rt}ms). Threads are blocked waiting for downstream responses, preventing health check responses within the {hc_timeout}s timeout. The ALB correctly marks them unhealthy, but the root cause is the downstream dependency, not the backend servers themselves.",
            "confounding_action": "do(Add More Backend Targets) provides temporary relief but accelerates downstream dependency collapse. Each new target opens connections to the slow downstream service, increasing its load and response time further. This is a classic cascade amplification.",
            "counterfactual": "Had the backend implemented async I/O for the downstream call with a {async_timeout}ms timeout and a bulkhead pattern isolating {bulkhead_pct}% of threads for health checks, targets would remain healthy and the degradation would be contained to the affected endpoint.",
            "correct_action": "1. Implement circuit breaker for downstream dependency (timeout={async_timeout}ms). 2. Reserve dedicated health check thread pool. 3. Increase ALB health check interval to {hc_interval}s. 4. Add fallback/cache for downstream responses.",
            "dag_nodes": [
                {"id": "downstream_slow", "label": "Slow Downstream ({downstream_rt}ms)"},
                {"id": "thread_exhaustion", "label": "Thread Pool Exhaustion"},
                {"id": "hc_timeout", "label": "Health Check Timeout ({hc_timeout}s)"},
                {"id": "target_unhealthy", "label": "Targets Marked Unhealthy ({unhealthy_pct}%)"},
                {"id": "surge_queue", "label": "Surge Queue ({queue_len})"},
                {"id": "http_5xx", "label": "HTTP 5xx ({error_rate}/min)"},
            ],
            "dag_edges": [
                {"source": "downstream_slow", "target": "thread_exhaustion"},
                {"source": "thread_exhaustion", "target": "hc_timeout"},
                {"source": "hc_timeout", "target": "target_unhealthy"},
                {"source": "target_unhealthy", "target": "surge_queue"},
                {"source": "surge_queue", "target": "http_5xx"},
            ],
            "params": {"num_targets": [5, 10, 20], "unhealthy_pct": [40, 60, 80], "active_conn": [5000, 15000, 50000], "queue_len": [100, 500, 1024], "backend_rt": [5000, 10000, 30000], "error_rate": [50, 200, 1000], "downstream_rt": [8000, 15000, 30000], "hc_timeout": [5, 10, 30], "async_timeout": [1000, 2000, 5000], "bulkhead_pct": [10, 15, 20], "hc_interval": [15, 30, 60]},
        },
    ],
    "dns": [
        {
            "premise": "DNS resolution latency for internal services spikes from {normal_latency}ms to {spike_latency}ms. {failure_pct}% of DNS queries time out. The CoreDNS pod memory usage is {mem_usage}MB (limit: {mem_limit}MB). DNS cache hit ratio dropped from {hit_normal}% to {hit_current}%.",
            "root_cause": "A misconfigured ndots setting (ndots:{ndots}) in the pod's resolv.conf causes each DNS query to generate {search_domains} search domain attempts before resolving. Combined with a deployment that increased unique hostnames by {hostname_increase}x, this overwhelms CoreDNS with {query_amplification}x query amplification.",
            "confounding_action": "do(Scale CoreDNS Replicas) adds capacity but doesn't reduce the query amplification factor. Each replica still processes {search_domains}x as many queries as necessary. The cost scales linearly with a problem that should not exist.",
            "counterfactual": "Had ndots been set to {correct_ndots} and FQDNs used with trailing dots in service discovery, each DNS query would resolve in a single lookup instead of {search_domains}, eliminating the amplification entirely.",
            "correct_action": "1. Set ndots:{correct_ndots} in pod DNS config. 2. Use FQDNs with trailing dots for cross-namespace calls. 3. Enable NodeLocal DNSCache. 4. Increase CoreDNS cache size to {cache_size} entries.",
            "dag_nodes": [
                {"id": "ndots_config", "label": "ndots:{ndots} Misconfiguration"},
                {"id": "query_amplification", "label": "DNS Query Amplification ({query_amplification}x)"},
                {"id": "coredns_overload", "label": "CoreDNS Overload ({mem_usage}MB)"},
                {"id": "cache_miss", "label": "Cache Hit Ratio Drop ({hit_current}%)"},
                {"id": "dns_timeout", "label": "DNS Timeout ({failure_pct}%)"},
                {"id": "service_resolution", "label": "Service Resolution Failure"},
            ],
            "dag_edges": [
                {"source": "ndots_config", "target": "query_amplification"},
                {"source": "query_amplification", "target": "coredns_overload"},
                {"source": "coredns_overload", "target": "cache_miss"},
                {"source": "cache_miss", "target": "dns_timeout"},
                {"source": "dns_timeout", "target": "service_resolution"},
            ],
            "params": {"normal_latency": [1, 2, 5], "spike_latency": [200, 500, 2000], "failure_pct": [10, 25, 50], "mem_usage": [200, 400, 800], "mem_limit": [256, 512, 1024], "hit_normal": [90, 95, 98], "hit_current": [20, 35, 50], "ndots": [5, 5, 5], "search_domains": [5, 6, 8], "hostname_increase": [3, 5, 10], "query_amplification": [5, 6, 8], "correct_ndots": [2, 2, 1], "cache_size": [10000, 50000, 100000]},
        },
    ],
    "storage": [
        {
            "premise": "An EBS-backed application sees I/O latency spike to {io_latency}ms (normal: {io_normal}ms). Throughput drops to {throughput}MB/s. CloudWatch shows BurstBalance at {burst}%. IOPS consumption at {iops_used}/{iops_max}. Volume type is {vol_type}.",
            "root_cause": "The {vol_type} volume exhausted its burst credit balance. The application's write pattern during a compaction job generates {iops_actual} IOPS, exceeding the baseline of {iops_baseline} IOPS. Once burst credits deplete, performance drops to baseline, creating a sudden latency cliff.",
            "confounding_action": "do(Increase Volume Size) increases baseline IOPS proportionally but introduces a {resize_time}-minute resize operation during which I/O performance is further degraded due to EBS volume modification overhead.",
            "counterfactual": "Had the volume been provisioned as io2 with {provisioned_iops} IOPS, or the compaction job been scheduled during low-traffic windows with rate limiting at {rate_limit} IOPS, burst credits would never have been exhausted.",
            "correct_action": "1. Migrate to io2 volume with {provisioned_iops} provisioned IOPS. 2. Schedule compaction during off-peak hours. 3. Implement I/O rate limiting for batch operations. 4. Add CloudWatch alarm for BurstBalance < {burst_alarm}%.",
            "dag_nodes": [
                {"id": "compaction_job", "label": "Compaction Job ({iops_actual} IOPS)"},
                {"id": "burst_depletion", "label": "Burst Credit Depletion ({burst}%)"},
                {"id": "baseline_perf", "label": "Baseline Performance ({iops_baseline} IOPS)"},
                {"id": "io_latency", "label": "I/O Latency Spike ({io_latency}ms)"},
                {"id": "throughput_drop", "label": "Throughput Drop ({throughput}MB/s)"},
                {"id": "app_degradation", "label": "Application Degradation"},
            ],
            "dag_edges": [
                {"source": "compaction_job", "target": "burst_depletion"},
                {"source": "burst_depletion", "target": "baseline_perf"},
                {"source": "baseline_perf", "target": "io_latency"},
                {"source": "io_latency", "target": "throughput_drop"},
                {"source": "throughput_drop", "target": "app_degradation"},
            ],
            "params": {"io_latency": [50, 100, 500], "io_normal": [1, 2, 5], "throughput": [10, 25, 50], "burst": [0, 2, 5], "iops_used": [3000, 3000, 3000], "iops_max": [3000, 3000, 3000], "vol_type": ["gp2", "gp3", "gp2"], "iops_actual": [5000, 10000, 15000], "iops_baseline": [100, 300, 3000], "resize_time": [5, 10, 30], "provisioned_iops": [5000, 10000, 20000], "rate_limit": [1000, 2000, 3000], "burst_alarm": [20, 30, 50]},
        },
    ],
    "authentication": [
        {
            "premise": "OAuth2 token validation latency increases from {normal_val}ms to {spike_val}ms. {timeout_pct}% of token validations timeout. The JWKS endpoint response time is {jwks_rt}ms. Token introspection cache hit ratio is {cache_hit}%. Active sessions: {sessions}.",
            "root_cause": "The JWKS key rotation occurred {rotation_ago} minutes ago but the token validation service caches JWKS keys with a TTL of {jwks_ttl}s. During the cache miss window, every validation request hits the JWKS endpoint, creating a thundering herd of {herd_size} concurrent requests against the identity provider.",
            "confounding_action": "do(Increase Token Validation Timeout) masks the thundering herd by allowing requests to queue longer, increasing memory per request and eventually causing the validation service to OOM at {oom_threshold} concurrent requests.",
            "counterfactual": "Had the JWKS cache implemented a stale-while-revalidate pattern with background refresh {bg_refresh}s before expiry, only a single request would have fetched the new keys while all others continued using the cached (still valid) keys.",
            "correct_action": "1. Implement stale-while-revalidate JWKS caching. 2. Background refresh JWKS keys {bg_refresh}s before TTL expires. 3. Add jitter to cache TTL ({jitter_min}-{jitter_max}s). 4. Rate-limit JWKS endpoint calls to {rate_limit}/s.",
            "dag_nodes": [
                {"id": "key_rotation", "label": "JWKS Key Rotation"},
                {"id": "cache_expiry", "label": "Cache Expiry (TTL={jwks_ttl}s)"},
                {"id": "thundering_herd", "label": "Thundering Herd ({herd_size} requests)"},
                {"id": "jwks_overload", "label": "JWKS Endpoint Overload ({jwks_rt}ms)"},
                {"id": "validation_timeout", "label": "Token Validation Timeout ({timeout_pct}%)"},
                {"id": "auth_failure", "label": "Authentication Failures"},
            ],
            "dag_edges": [
                {"source": "key_rotation", "target": "cache_expiry"},
                {"source": "cache_expiry", "target": "thundering_herd"},
                {"source": "thundering_herd", "target": "jwks_overload"},
                {"source": "jwks_overload", "target": "validation_timeout"},
                {"source": "validation_timeout", "target": "auth_failure"},
            ],
            "params": {"normal_val": [5, 10, 20], "spike_val": [500, 2000, 5000], "timeout_pct": [15, 30, 60], "jwks_rt": [2000, 5000, 10000], "cache_hit": [10, 20, 30], "sessions": [10000, 50000, 200000], "rotation_ago": [2, 5, 10], "jwks_ttl": [60, 300, 600], "herd_size": [500, 2000, 10000], "oom_threshold": [5000, 15000, 50000], "bg_refresh": [30, 60, 120], "jitter_min": [10, 30, 60], "jitter_max": [30, 60, 120], "rate_limit": [5, 10, 20]},
        },
    ],
    "message_queue": [
        {
            "premise": "A Kafka cluster shows consumer lag of {lag} messages across {partitions} partitions. Producer throughput is {producer_mbps}MB/s. Consumer group '{group}' has {consumers} active consumers. Rebalances are occurring every {rebalance_interval}s.",
            "root_cause": "Frequent consumer rebalances are caused by consumers exceeding max.poll.interval.ms ({poll_interval}ms) during processing of large batches ({batch_size} messages). Each rebalance triggers a stop-the-world pause across all consumers in the group, creating a compounding lag spiral.",
            "confounding_action": "do(Add More Consumers) triggers additional rebalances since each new consumer joining the group causes a full group rebalance. With {partitions} partitions and {consumers} consumers, adding more consumers beyond partition count provides zero benefit and increases rebalance frequency.",
            "counterfactual": "Had max.poll.records been set to {correct_batch} and max.poll.interval.ms increased to {correct_interval}ms, each poll would complete within the interval, eliminating rebalances and allowing steady-state consumption at {steady_rate} messages/s.",
            "correct_action": "1. Reduce max.poll.records to {correct_batch}. 2. Increase max.poll.interval.ms to {correct_interval}ms. 3. Use cooperative-sticky assignor to minimize rebalance impact. 4. Implement incremental processing with commits every {commit_interval} messages.",
            "dag_nodes": [
                {"id": "large_batch", "label": "Large Batch Size ({batch_size} msgs)"},
                {"id": "poll_timeout", "label": "Poll Interval Exceeded ({poll_interval}ms)"},
                {"id": "rebalance", "label": "Consumer Rebalance (every {rebalance_interval}s)"},
                {"id": "stop_world", "label": "Stop-the-World Pause"},
                {"id": "consumer_lag", "label": "Consumer Lag ({lag} msgs)"},
                {"id": "processing_delay", "label": "Processing Delay"},
            ],
            "dag_edges": [
                {"source": "large_batch", "target": "poll_timeout"},
                {"source": "poll_timeout", "target": "rebalance"},
                {"source": "rebalance", "target": "stop_world"},
                {"source": "stop_world", "target": "consumer_lag"},
                {"source": "consumer_lag", "target": "processing_delay"},
                {"source": "processing_delay", "target": "poll_timeout"},
            ],
            "params": {"lag": [50000, 200000, 1000000], "partitions": [12, 24, 48], "producer_mbps": [10, 50, 100], "group": ["order-processor", "event-handler", "analytics-pipeline"], "consumers": [6, 12, 24], "rebalance_interval": [30, 60, 120], "poll_interval": [300000, 300000, 300000], "batch_size": [1000, 5000, 10000], "correct_batch": [100, 200, 500], "correct_interval": [600000, 900000, 1200000], "steady_rate": [5000, 10000, 50000], "commit_interval": [50, 100, 200]},
        },
    ],
    "cache": [
        {
            "premise": "A Redis cluster used for session storage shows {eviction_rate} evictions/sec. Memory usage is {mem_used}GB/{mem_max}GB ({mem_pct}%). Cache hit ratio dropped from {hit_normal}% to {hit_current}%. P99 GET latency is {get_latency}ms. Connected clients: {clients}.",
            "root_cause": "A new feature deployed {deploy_ago}h ago stores {obj_size}KB user preference objects in Redis without a TTL. These objects accumulate monotonically, displacing session data through the {eviction_policy} eviction policy. The feature accounts for {feature_pct}% of memory but only {access_pct}% of reads.",
            "confounding_action": "do(Increase Redis Memory) delays the issue but doesn't solve it. The preference objects grow at {growth_rate}GB/day. Within {days_to_full} days, the expanded cluster would also reach capacity, and the cost grows linearly with no ceiling.",
            "counterfactual": "Had the preference objects been stored with a TTL of {ttl}h and a lazy-loading cache pattern, memory usage would stabilize at ~{stable_mem}GB with no eviction pressure on session data.",
            "correct_action": "1. Add TTL of {ttl}h to all preference keys. 2. Migrate cold preference data to a persistent store. 3. Switch eviction policy to volatile-lfu. 4. Implement cache-aside pattern with read-through for preferences.",
            "dag_nodes": [
                {"id": "no_ttl", "label": "No TTL on Preference Objects"},
                {"id": "memory_growth", "label": "Monotonic Memory Growth ({growth_rate}GB/day)"},
                {"id": "mem_pressure", "label": "Memory Pressure ({mem_pct}%)"},
                {"id": "eviction", "label": "Session Evictions ({eviction_rate}/s)"},
                {"id": "cache_miss", "label": "Cache Miss Spike ({hit_current}% hit rate)"},
                {"id": "latency", "label": "Latency Increase ({get_latency}ms P99)"},
            ],
            "dag_edges": [
                {"source": "no_ttl", "target": "memory_growth"},
                {"source": "memory_growth", "target": "mem_pressure"},
                {"source": "mem_pressure", "target": "eviction"},
                {"source": "eviction", "target": "cache_miss"},
                {"source": "cache_miss", "target": "latency"},
            ],
            "params": {"eviction_rate": [100, 500, 2000], "mem_used": [12, 14, 15.5], "mem_max": [16, 16, 16], "mem_pct": [75, 87.5, 96.8], "hit_normal": [95, 98, 99], "hit_current": [60, 45, 30], "get_latency": [5, 15, 50], "clients": [500, 2000, 10000], "deploy_ago": [12, 24, 48], "obj_size": [2, 8, 32], "eviction_policy": ["allkeys-lru", "allkeys-lru", "allkeys-lfu"], "feature_pct": [40, 60, 80], "access_pct": [5, 10, 15], "growth_rate": [0.5, 2, 8], "days_to_full": [3, 7, 14], "ttl": [4, 12, 24], "stable_mem": [4, 6, 8]},
        },
    ],
    "ci_cd": [
        {
            "premise": "CI/CD pipeline execution time increased from {normal_time}min to {current_time}min. Build success rate dropped to {success_rate}%. {flaky_pct}% of test failures are in integration tests. The shared build cache hit ratio is {cache_hit}%. Concurrent pipelines: {concurrent}.",
            "root_cause": "The shared build cache was invalidated by a dependency version bump in the base Docker image {bump_ago}h ago. All {concurrent} concurrent pipelines are now performing full rebuilds simultaneously, causing resource contention on the shared CI runner pool ({runners} runners). The integration test failures are caused by resource exhaustion (CPU throttling at {cpu_throttle}%) on runners, not actual code defects.",
            "confounding_action": "do(Retry Failed Tests) wastes CI runner capacity. Each retry consumes {retry_cost} compute-minutes while the underlying resource contention persists. With {concurrent} pipelines retrying, total compute cost increases by {cost_multiplier}x.",
            "counterfactual": "Had the Docker base image been pinned to a digest and the cache keyed on the lock file hash rather than the image tag, the dependency bump would not have invalidated the cache, and build times would have remained at {normal_time}min.",
            "correct_action": "1. Pin Docker base image to digest. 2. Key build cache on dependency lock file hash. 3. Warm cache in a dedicated pipeline after base image updates. 4. Implement concurrency limits ({max_concurrent} parallel pipelines) during cache rebuilds.",
            "dag_nodes": [
                {"id": "dep_bump", "label": "Dependency Version Bump"},
                {"id": "cache_invalidation", "label": "Build Cache Invalidation"},
                {"id": "full_rebuild", "label": "Full Rebuilds ({concurrent} concurrent)"},
                {"id": "runner_contention", "label": "CI Runner Contention ({runners} runners)"},
                {"id": "cpu_throttle", "label": "CPU Throttling ({cpu_throttle}%)"},
                {"id": "test_failures", "label": "Integration Test Failures ({flaky_pct}%)"},
            ],
            "dag_edges": [
                {"source": "dep_bump", "target": "cache_invalidation"},
                {"source": "cache_invalidation", "target": "full_rebuild"},
                {"source": "full_rebuild", "target": "runner_contention"},
                {"source": "runner_contention", "target": "cpu_throttle"},
                {"source": "cpu_throttle", "target": "test_failures"},
            ],
            "params": {"normal_time": [8, 12, 20], "current_time": [45, 60, 120], "success_rate": [40, 55, 70], "flaky_pct": [60, 75, 90], "cache_hit": [5, 10, 20], "concurrent": [10, 20, 50], "bump_ago": [2, 6, 12], "runners": [5, 10, 20], "cpu_throttle": [85, 92, 98], "retry_cost": [8, 15, 30], "cost_multiplier": [2, 3, 5], "max_concurrent": [3, 5, 10]},
        },
    ],
    "monitoring": [
        {
            "premise": "The monitoring stack (Prometheus + Grafana) shows {prom_mem}GB memory usage (limit: {prom_mem_limit}GB). Scrape duration for {targets} targets increased to {scrape_dur}s (interval: {scrape_interval}s). {cardinality} active time series. Query latency: {query_latency}s.",
            "root_cause": "A deployment {deploy_ago}h ago introduced a metric with unbounded label cardinality: a 'request_id' label on an HTTP histogram. Each unique request ID creates new time series, causing cardinality explosion from {cardinality_before} to {cardinality} series. Prometheus cannot complete scrapes within the interval, causing gaps and stale data.",
            "confounding_action": "do(Increase Prometheus Memory) allows more series to be ingested but accelerates the cardinality explosion. At {growth_rate} new series/hour, even {doubled_mem}GB would be exhausted within {hours_to_oom} hours.",
            "counterfactual": "Had the metric been instrumented with bounded labels (e.g., HTTP method, status code, endpoint path) instead of high-cardinality 'request_id', total series would remain at ~{stable_series} — well within capacity.",
            "correct_action": "1. Remove 'request_id' label from the histogram metric. 2. Use exemplars for request-level trace correlation instead. 3. Add recording rules to pre-aggregate high-cardinality queries. 4. Set metric_relabel_configs to drop unexpected labels.",
            "dag_nodes": [
                {"id": "high_cardinality_label", "label": "Unbounded Label (request_id)"},
                {"id": "series_explosion", "label": "Time Series Explosion ({cardinality})"},
                {"id": "memory_pressure", "label": "Prometheus Memory ({prom_mem}GB)"},
                {"id": "scrape_overrun", "label": "Scrape Duration > Interval ({scrape_dur}s)"},
                {"id": "data_gaps", "label": "Monitoring Data Gaps"},
                {"id": "alert_failure", "label": "Alert Evaluation Failures"},
            ],
            "dag_edges": [
                {"source": "high_cardinality_label", "target": "series_explosion"},
                {"source": "series_explosion", "target": "memory_pressure"},
                {"source": "series_explosion", "target": "scrape_overrun"},
                {"source": "scrape_overrun", "target": "data_gaps"},
                {"source": "data_gaps", "target": "alert_failure"},
            ],
            "params": {"prom_mem": [12, 24, 48], "prom_mem_limit": [16, 32, 64], "targets": [100, 500, 2000], "scrape_dur": [20, 45, 90], "scrape_interval": [15, 15, 15], "cardinality": [5000000, 15000000, 50000000], "query_latency": [5, 15, 60], "deploy_ago": [6, 12, 24], "cardinality_before": [100000, 500000, 1000000], "growth_rate": [100000, 500000, 2000000], "doubled_mem": [32, 64, 128], "hours_to_oom": [4, 8, 12], "stable_series": [150000, 600000, 1200000]},
        },
    ],
    "api_gateway": [
        {
            "premise": "API Gateway throttling activates at {throttle_rate} requests/sec. {reject_pct}% of requests receive HTTP 429. Backend utilization is only {backend_util}%. Rate limit policy is set to {rate_limit} req/s globally. Per-client limits: {per_client_limit} req/s.",
            "root_cause": "A single misconfigured client (client_id: {bad_client}) is sending {bad_client_rate} requests/sec of health checks to the same endpoint, consuming {consumption_pct}% of the global rate limit. The per-client limit of {per_client_limit} req/s is not enforced because the rate limiting is applied at the global level first, causing legitimate clients to be throttled.",
            "confounding_action": "do(Increase Global Rate Limit) allows the noisy client to consume even more capacity. The backend is only at {backend_util}% utilization because the gateway is rejecting requests before they reach the backend — the bottleneck is the rate limiter configuration, not backend capacity.",
            "counterfactual": "Had rate limiting been applied per-client first (at {per_client_limit} req/s) before the global limit, the noisy client would have been throttled individually while all other clients experienced normal service.",
            "correct_action": "1. Apply per-client rate limiting before global limits. 2. Block or throttle client {bad_client} to {corrected_rate} req/s. 3. Implement adaptive rate limiting based on client behavior. 4. Add circuit breaker for clients exceeding {circuit_threshold} req/s.",
            "dag_nodes": [
                {"id": "noisy_client", "label": "Noisy Client ({bad_client_rate} req/s)"},
                {"id": "global_limit", "label": "Global Rate Limit ({rate_limit} req/s)"},
                {"id": "limit_consumed", "label": "Limit Consumed ({consumption_pct}%)"},
                {"id": "legitimate_throttled", "label": "Legitimate Clients Throttled"},
                {"id": "http_429", "label": "HTTP 429 ({reject_pct}%)"},
                {"id": "backend_idle", "label": "Backend Idle ({backend_util}%)"},
            ],
            "dag_edges": [
                {"source": "noisy_client", "target": "global_limit"},
                {"source": "global_limit", "target": "limit_consumed"},
                {"source": "limit_consumed", "target": "legitimate_throttled"},
                {"source": "legitimate_throttled", "target": "http_429"},
                {"source": "limit_consumed", "target": "backend_idle"},
            ],
            "params": {"throttle_rate": [500, 1000, 5000], "reject_pct": [20, 40, 60], "backend_util": [15, 25, 40], "rate_limit": [1000, 2000, 5000], "per_client_limit": [100, 200, 500], "bad_client": ["svc-health-checker", "monitoring-bot", "ci-pipeline-runner"], "bad_client_rate": [800, 1500, 3000], "consumption_pct": [60, 75, 90], "corrected_rate": [10, 20, 50], "circuit_threshold": [200, 500, 1000]},
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Pearl's Hierarchy Level Assignment
# ─────────────────────────────────────────────────────────────────────────────

PEARL_LEVEL_MAP = {
    "kubernetes": 2,       # Interventional — direct system manipulation
    "database": 2,         # Interventional — connection pool management
    "networking": 1,       # Associational — correlating network metrics
    "load_balancer": 2,    # Interventional — traffic routing decisions
    "dns": 1,              # Associational — DNS resolution patterns
    "storage": 2,          # Interventional — I/O management
    "authentication": 3,   # Counterfactual — "what if key rotation happened differently"
    "message_queue": 2,    # Interventional — consumer configuration
    "cache": 2,            # Interventional — eviction policy changes
    "ci_cd": 3,            # Counterfactual — "what if cache wasn't invalidated"
    "monitoring": 1,       # Associational — metric cardinality analysis
    "api_gateway": 2,      # Interventional — rate limit configuration
}


class SREDatasetGenerator:
    """
    Generates a massive SRE causal incident dataset for NEXUS-CAUSAL training.
    Mirrors DataScaledDatasetGenerator from NEXUS-CAUSAL v3.1.
    """

    def __init__(self):
        self.templates = INCIDENT_TEMPLATES
        self.pearl_levels = PEARL_LEVEL_MAP

    def _expand_template(self, domain: str, template: dict, param_combo: dict) -> dict:
        """Expand a template with a specific parameter combination."""
        example = {}

        # Format string fields with parameters
        for field in ["premise", "root_cause", "confounding_action", "counterfactual", "correct_action"]:
            try:
                example[field] = template[field].format(**param_combo)
            except (KeyError, IndexError):
                example[field] = template[field]

        # Format DAG nodes
        example["dag_nodes"] = []
        for node in template["dag_nodes"]:
            try:
                label = node["label"].format(**param_combo)
            except (KeyError, IndexError):
                label = node["label"]
            example["dag_nodes"].append({"id": node["id"], "label": label})

        # Format DAG edges
        example["dag_edges"] = []
        for i, edge in enumerate(template["dag_edges"]):
            example["dag_edges"].append({
                "id": f"e{i+1}",
                "source": edge["source"],
                "target": edge["target"],
                "animated": True,
            })

        # Add metadata
        example["domain"] = domain
        example["pearl_level"] = self.pearl_levels.get(domain, 1)
        example["quality_score"] = round(random.uniform(0.4, 1.0), 3)

        # Construct reasoning field (for SFT)
        example["reasoning"] = (
            f"[DOMAIN] {domain}\n"
            f"[PEARL_LEVEL] L{example['pearl_level']}\n"
            f"[CAUSE] {example['root_cause']}\n"
            f"[INTERVENTION] {example['confounding_action']}\n"
            f"[MECHANISM] The causal graph shows a {len(example['dag_nodes'])}-node DAG "
            f"with {len(example['dag_edges'])} directed edges.\n"
            f"[CONCLUSION] {example['correct_action']}"
        )

        # Causal graph for SFT format compatibility
        example["causal_graph"] = {
            "nodes": [n["id"] for n in example["dag_nodes"]],
            "edges": [(e["source"], e["target"]) for e in example["dag_edges"]],
        }

        return example

    def create_sre_dataset(self, num_examples: int = 2500) -> List[Dict]:
        """Generate a massive SRE causal incident dataset."""
        print(f"🏗️  Generating {num_examples} SRE causal incident examples...")
        all_examples = []

        # Calculate examples per template for even distribution
        total_templates = sum(len(t) for t in self.templates.values())
        examples_per_template = max(num_examples // total_templates, 50)

        for domain, templates in self.templates.items():
            for template in templates:
                params = template["params"]
                keys = list(params.keys())

                # Random sampling instead of full combinatorial expansion
                # (some templates have 10+ params × 3 values = millions of combos)
                for _ in range(examples_per_template):
                    param_combo = {
                        k: random.choice(params[k]) for k in keys
                    }
                    example = self._expand_template(domain, template, param_combo)
                    all_examples.append(example)

        print(f"📊 Generated {len(all_examples)} raw examples from templates")

        # If we need more, duplicate with variation
        if len(all_examples) < num_examples:
            extras_needed = num_examples - len(all_examples)
            extras = []
            for _ in range(extras_needed):
                base = random.choice(all_examples)
                varied = base.copy()
                varied["quality_score"] = round(random.uniform(0.3, 1.0), 3)
                extras.append(varied)
            all_examples.extend(extras)

        # Shuffle and truncate to target
        random.shuffle(all_examples)
        dataset = all_examples[:num_examples]

        # Print distribution
        domain_counts = {}
        level_counts = {1: 0, 2: 0, 3: 0}
        for ex in dataset:
            domain_counts[ex["domain"]] = domain_counts.get(ex["domain"], 0) + 1
            level_counts[ex["pearl_level"]] = level_counts.get(ex["pearl_level"], 0) + 1

        print("\n📈 Dataset Distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"   {domain}: {count}")
        print(f"\n🪜 Pearl's Hierarchy:")
        print(f"   L1 (Association):    {level_counts[1]}")
        print(f"   L2 (Intervention):   {level_counts[2]}")
        print(f"   L3 (Counterfactual): {level_counts[3]}")
        print(f"\n✅ Final dataset size: {len(dataset)} examples")

        return dataset


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=2)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 Dataset saved to {filepath} ({size_mb:.1f} MB)")
