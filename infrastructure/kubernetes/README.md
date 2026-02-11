# Kubernetes Deployment Guide for ShikshaSetu

This directory contains Kubernetes manifests and Kustomize overlays for deploying ShikshaSetu in different environments.

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
   docker build -t shiksha-setu/api:v1.0.0 -f Dockerfile .
   docker build -t shiksha-setu/worker:v1.0.0 -f Dockerfile.worker .
   
   # For production, push to your registry
   docker tag shiksha-setu/api:v1.0.0 your-registry.io/shiksha-setu/api:v1.0.0
   docker push your-registry.io/shiksha-setu/api:v1.0.0
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
docker build -t shiksha-setu/api:latest -f Dockerfile .
docker build -t shiksha-setu/worker:latest -f Dockerfile.worker .

# Load into minikube
minikube image load shiksha-setu/api:latest
minikube image load shiksha-setu/worker:latest
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
kubectl create secret generic shiksha-secrets \
  --from-env-file=secrets.env \
  --namespace=shiksha-setu \
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
kubectl get all -n shiksha-setu

# Watch pods come up
kubectl get pods -n shiksha-setu -w
```

### 5. Run Database Migrations

```bash
# Wait for postgres to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n shiksha-setu --timeout=300s

# Run migrations
kubectl exec -it deployment/fastapi -n shiksha-setu -- alembic upgrade head
```

### 6. Access the Application

```bash
# Get the service URL
minikube service fastapi -n shiksha-setu --url

# Or port-forward
kubectl port-forward svc/fastapi 8000:8000 -n shiksha-setu

# Access at http://localhost:8000
```

## Environment-Specific Deployments

### Development

```bash
# Deploy dev environment
kubectl apply -k overlays/dev/

# Access dev namespace
kubectl get all -n shiksha-setu-dev
```

### Staging

```bash
# Set environment variables for secrets
export STAGING_DB_PASSWORD="staging-password"
export STAGING_SECRET_KEY="staging-secret"

# Deploy staging
kubectl apply -k overlays/staging/

# Check status
kubectl get all -n shiksha-setu-staging
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
kubectl get all -n shiksha-setu-prod
kubectl get hpa -n shiksha-setu-prod
```

## Monitoring and Logs

### View Logs

```bash
# FastAPI logs
kubectl logs -f deployment/fastapi -n shiksha-setu

# Celery worker logs
kubectl logs -f deployment/celery-worker -n shiksha-setu

# Specific pod
kubectl logs -f <pod-name> -n shiksha-setu
```

### Check Health

```bash
# FastAPI health endpoint
kubectl port-forward svc/fastapi 8000:8000 -n shiksha-setu
curl http://localhost:8000/health

# Check HPA status
kubectl get hpa -n shiksha-setu

# Check resource usage
kubectl top pods -n shiksha-setu
kubectl top nodes
```

### Debug Pods

```bash
# Shell into FastAPI pod
kubectl exec -it deployment/fastapi -n shiksha-setu -- /bin/bash

# Shell into worker pod
kubectl exec -it deployment/celery-worker -n shiksha-setu -- /bin/bash

# Check database connection
kubectl exec -it deployment/postgres -n shiksha-setu -- psql -U postgres -d shiksha_setu
```

## Scaling

### Manual Scaling

```bash
# Scale FastAPI
kubectl scale deployment fastapi --replicas=5 -n shiksha-setu

# Scale workers
kubectl scale deployment celery-worker --replicas=10 -n shiksha-setu
```

### Horizontal Pod Autoscaling

HPAs are configured in `deployment.yaml`:
- **FastAPI**: 2-10 replicas (70% CPU, 80% memory)
- **Celery Workers**: 3-20 replicas (75% CPU, 85% memory)

Check HPA status:
```bash
kubectl get hpa -n shiksha-setu
kubectl describe hpa fastapi-hpa -n shiksha-setu
```

## Persistent Storage

### Check PVCs

```bash
kubectl get pvc -n shiksha-setu
kubectl describe pvc data-pvc -n shiksha-setu
```

### Backup Data

```bash
# Backup PostgreSQL
kubectl exec deployment/postgres -n shiksha-setu -- \
  pg_dump -U postgres shiksha_setu > backup.sql

# Backup data volume
kubectl cp shiksha-setu/fastapi-<pod-id>:/app/data ./data-backup
```

## Networking

### Ingress

The ingress is configured in `ingress.yaml`. Update the host to your domain:

```bash
# Edit ingress
kubectl edit ingress shiksha-ingress -n shiksha-setu

# Check ingress
kubectl get ingress -n shiksha-setu
kubectl describe ingress shiksha-ingress -n shiksha-setu
```

For Minikube:
```bash
# Get minikube IP
minikube ip

# Add to /etc/hosts
echo "$(minikube ip) api.shiksha-setu.local" | sudo tee -a /etc/hosts
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
kubectl delete namespace shiksha-setu
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
kubectl describe pod <pod-name> -n shiksha-setu

# Check events
kubectl get events -n shiksha-setu --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n shiksha-setu --previous
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -n shiksha-setu -- \
  psql -h postgres -U postgres -d shiksha_setu

# Check service endpoints
kubectl get endpoints -n shiksha-setu
```

### Storage Issues

```bash
# Check PV/PVC status
kubectl get pv,pvc -n shiksha-setu

# Describe PVC for events
kubectl describe pvc data-pvc -n shiksha-setu
```

### Image Pull Issues

```bash
# For local images in Minikube
minikube image ls | grep shiksha-setu

# Re-load images if needed
minikube image load shiksha-setu/api:latest
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
