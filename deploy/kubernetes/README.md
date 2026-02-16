# Kubernetes Deployment Guide for Oryon

This directory contains Kubernetes manifests and Kustomize overlays for deploying Oryon in different environments.

## Architecture

The application consists of:
- **FastAPI Service**: Main API server (2+ replicas with HPA)
- **Celery Workers**: Background task processors (3+ replicas with HPA)
- **PostgreSQL**: Database (StatefulSet with persistent storage)
- **Redis**: Cache and message broker (StatefulSet with persistent storage)

## Prerequisites

1. **Kubernetes Cluster**: 
   - Local: Minikube, Kind, Docker Desktop
   - Cloud: GKE, EKS, AKS, or DigitalOcean Kubernetes

2. **Tools**:
   ```bash
   # kubectl
   brew install kubectl
   
   # kustomize (optional, kubectl has built-in kustomize)
   brew install kustomize
   
   # For local development - minikube
   brew install minikube
   
   # Or kind
   brew install kind
   ```

3. **Docker Images**:
   Build your images locally or push to a registry:
   ```bash
   # Build images with semantic versioning (recommended)
   docker build -t oryon-ai/api:v1.0.0 -f Dockerfile .
   docker build -t oryon-ai/worker:v1.0.0 -f Dockerfile.worker .
   
   # For production, push to your registry
   docker tag oryon-ai/api:v1.0.0 your-registry.io/oryon-ai/api:v1.0.0
   docker push your-registry.io/oryon-ai/api:v1.0.0
   ```
   
   **Note**: The base `kustomization.yaml` uses semantic versioning (v1.0.0) instead of `:latest` tags for better reproducibility and rollback capabilities.

## Quick Start (Minikube)

### 1. Start Minikube

```bash
# Start with sufficient resources
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable storage-provisioner
```

### 2. Load Docker Images (for local development)

```bash
# Build images
docker build -t oryon-ai/api:latest -f Dockerfile .
docker build -t oryon-ai/worker:latest -f Dockerfile.worker .

# Load into minikube
minikube image load oryon-ai/api:latest
minikube image load oryon-ai/worker:latest
```

### 3. Configure Secrets

```bash
# Create secrets file (don't commit this!)
cat > secrets.env <<EOF
POSTGRES_PASSWORD=your-secure-password
HUGGINGFACE_API_KEY=your-hf-api-key
FLASK_SECRET_KEY=your-flask-secret
EOF

# Create the secret
kubectl create secret generic oryon-secrets \
  --from-env-file=secrets.env \
  --namespace=oryon-ai \
  --dry-run=client -o yaml | kubectl apply -f -

# Clean up
rm secrets.env
```

### 4. Deploy

```bash
# Deploy everything
kubectl apply -f deployment.yaml

# Or use kustomize for base deployment
kubectl apply -k .

# Check status
kubectl get all -n oryon-ai

# Watch pods come up
kubectl get pods -n oryon-ai -w
```

### 5. Run Database Migrations

```bash
# Wait for postgres to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n oryon-ai --timeout=300s

# Run migrations
kubectl exec -it deployment/fastapi -n oryon-ai -- alembic upgrade head
```

### 6. Access the Application

```bash
# Get the service URL
minikube service fastapi -n oryon-ai --url

# Or port-forward
kubectl port-forward svc/fastapi 8000:8000 -n oryon-ai

# Access at http://localhost:8000
```

## Environment-Specific Deployments

### Development

```bash
# Deploy dev environment
kubectl apply -k overlays/dev/

# Access dev namespace
kubectl get all -n oryon-ai-dev
```

### Staging

```bash
# Set environment variables for secrets
export STAGING_DB_PASSWORD="staging-password"
export STAGING_SECRET_KEY="staging-secret"

# Deploy staging
kubectl apply -k overlays/staging/

# Check status
kubectl get all -n oryon-ai-staging
```

### Production

```bash
# For production, use external secrets management
# Example with Sealed Secrets:
kubeseal --format=yaml < secrets.yaml > sealed-secrets.yaml
kubectl apply -f sealed-secrets.yaml

# Deploy production
kubectl apply -k overlays/prod/

# Verify deployment
kubectl get all -n oryon-ai-prod
kubectl get hpa -n oryon-ai-prod
```

## Monitoring and Logs

### View Logs

```bash
# FastAPI logs
kubectl logs -f deployment/fastapi -n oryon-ai

# Celery worker logs
kubectl logs -f deployment/celery-worker -n oryon-ai

# Specific pod
kubectl logs -f <pod-name> -n oryon-ai
```

### Check Health

```bash
# FastAPI health endpoint
kubectl port-forward svc/fastapi 8000:8000 -n oryon-ai
curl http://localhost:8000/health

# Check HPA status
kubectl get hpa -n oryon-ai

# Check resource usage
kubectl top pods -n oryon-ai
kubectl top nodes
```

### Debug Pods

```bash
# Shell into FastAPI pod
kubectl exec -it deployment/fastapi -n oryon-ai -- /bin/bash

# Shell into worker pod
kubectl exec -it deployment/celery-worker -n oryon-ai -- /bin/bash

# Check database connection
kubectl exec -it deployment/postgres -n oryon-ai -- psql -U postgres -d oryon
```

## Scaling

### Manual Scaling

```bash
# Scale FastAPI
kubectl scale deployment fastapi --replicas=5 -n oryon-ai

# Scale workers
kubectl scale deployment celery-worker --replicas=10 -n oryon-ai
```

### Horizontal Pod Autoscaling

HPAs are configured in `deployment.yaml`:
- **FastAPI**: 2-10 replicas (70% CPU, 80% memory)
- **Celery Workers**: 3-20 replicas (75% CPU, 85% memory)

Check HPA status:
```bash
kubectl get hpa -n oryon-ai
kubectl describe hpa fastapi-hpa -n oryon-ai
```

## Persistent Storage

### Check PVCs

```bash
kubectl get pvc -n oryon-ai
kubectl describe pvc data-pvc -n oryon-ai
```

### Backup Data

```bash
# Backup PostgreSQL
kubectl exec deployment/postgres -n oryon-ai -- \
  pg_dump -U postgres oryon > backup.sql

# Backup data volume
kubectl cp oryon-ai/fastapi-<pod-id>:/app/data ./data-backup
```

## Networking

### Ingress

The ingress is configured in `ingress.yaml`. Update the host to your domain:

```bash
# Edit ingress
kubectl edit ingress oryon-ingress -n oryon-ai

# Check ingress
kubectl get ingress -n oryon-ai
kubectl describe ingress oryon-ingress -n oryon-ai
```

For Minikube:
```bash
# Get minikube IP
minikube ip

# Add to /etc/hosts
echo "$(minikube ip) api.oryon-ai.local" | sudo tee -a /etc/hosts
```

### Network Policies

Network policies are defined in `network-policy.yaml` to:
- Restrict database access to only FastAPI and workers
- Allow FastAPI to access external APIs (HuggingFace)
- Isolate services for security

Apply policies:
```bash
kubectl apply -f network-policy.yaml
```

## Cleanup

### Delete Namespace (removes everything)

```bash
kubectl delete namespace oryon-ai
```

### Delete Specific Resources

```bash
# Delete deployments
kubectl delete -f deployment.yaml

# Or with kustomize
kubectl delete -k .
```

### Stop Minikube

```bash
minikube stop
minikube delete  # Completely remove cluster
```

## Production Checklist

- [ ] Use external secrets management (Sealed Secrets, External Secrets Operator)
- [ ] Configure persistent volume backups
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK, Loki)
- [ ] Use TLS certificates (cert-manager with Let's Encrypt)
- [ ] Configure resource quotas and limits
- [ ] Set up CI/CD pipeline (GitHub Actions, GitLab CI)
- [ ] Enable pod security policies
- [ ] Configure network policies
- [ ] Set up alerting (PagerDuty, Opsgenie)
- [ ] Use image scanning (Trivy, Snyk)
- [ ] Configure pod disruption budgets
- [ ] Test disaster recovery procedures

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n oryon-ai

# Check events
kubectl get events -n oryon-ai --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n oryon-ai --previous
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -n oryon-ai -- \
  psql -h postgres -U postgres -d oryon

# Check service endpoints
kubectl get endpoints -n oryon-ai
```

### Storage Issues

```bash
# Check PV/PVC status
kubectl get pv,pvc -n oryon-ai

# Describe PVC for events
kubectl describe pvc data-pvc -n oryon-ai
```

### Image Pull Issues

```bash
# For local images in Minikube
minikube image ls | grep oryon-ai

# Re-load images if needed
minikube image load oryon-ai/api:latest
```

## Configuration Management

For detailed information about configuring the Kubernetes deployment, including:
- Environment-specific variable substitution (domains, AWS account IDs)
- Secret management strategies
- Image version management
- Network policies and security

See the **[Configuration Guide](CONFIGURATION.md)** for complete details.

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)

## Support

For issues or questions, refer to:
- Project documentation in `/docs`
- API documentation at `/docs` endpoint when running
- Technical documentation in `/docs/technical`
