# Kubernetes Setup Complete ✅

## 🎉 What's Installed

- **kubectl** v1.34.2 - Kubernetes command-line tool
- **minikube** v1.37.0 - Local Kubernetes cluster
- **kustomize** v5.7.1 - Kubernetes configuration management

## 🚀 Cluster Status

Your minikube cluster is **running** with:
- **CPUs**: 2
- **Memory**: 6GB
- **Disk**: 30GB
- **Kubernetes**: v1.34.0

### Enabled Addons
- ✅ **Ingress** - Route external traffic to services
- ✅ **Metrics Server** - Resource usage monitoring
- ✅ **Dashboard** - Web UI for cluster management
- ✅ **Storage Provisioner** - Dynamic volume provisioning

## 📋 Quick Commands

### Cluster Management
```bash
# Check cluster status
minikube status

# Stop cluster (saves state)
minikube stop

# Start cluster again
minikube start

# Delete cluster completely
minikube delete

# Access Kubernetes dashboard
minikube dashboard
```

### kubectl Basics
```bash
# View cluster info
kubectl cluster-info

# Get all resources
kubectl get all -A

# Get pods in all namespaces
kubectl get pods -A

# Get nodes
kubectl get nodes

# Describe a resource
kubectl describe pod <pod-name>

# View logs
kubectl logs <pod-name>

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/bash
```

### Context & Namespace
```bash
# View current context
kubectl config current-context

# List all contexts
kubectl config get-contexts

# Switch namespace
kubectl config set-context --current --namespace=oryon-ai

# View current namespace
kubectl config view --minify | grep namespace
```

## 🔧 Deploy Oryon

### Quick Deploy
```bash
# From project root
./scripts/deploy-k8s.sh dev
```

### Manual Deploy
```bash
# Build images
docker build -t oryon-ai/api:latest -f Dockerfile .
docker build -t oryon-ai/worker:latest -f Dockerfile.worker .

# Load images to minikube
minikube image load oryon-ai/api:latest
minikube image load oryon-ai/worker:latest

# Create namespace
kubectl create namespace oryon-ai

# Create secrets
kubectl create secret generic oryon-secrets \
  --from-literal=POSTGRES_PASSWORD=yourpassword \
  --from-literal=HUGGINGFACE_API_KEY=yourkey \
  --from-literal=FLASK_SECRET_KEY=yoursecret \
  --namespace=oryon-ai

# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get all -n oryon-ai

# Access application
kubectl port-forward svc/fastapi 8000:8000 -n oryon-ai
# Visit: http://localhost:8000
```

## 🌐 Networking

### Port Forwarding
```bash
# Forward service port to localhost
kubectl port-forward svc/fastapi 8000:8000 -n oryon-ai

# Forward specific pod
kubectl port-forward pod/fastapi-xxx 8000:8000 -n oryon-ai
```

### Ingress Access
```bash
# Get minikube IP
minikube ip

# Add to /etc/hosts
echo "$(minikube ip) api.oryon-ai.local" | sudo tee -a /etc/hosts

# Start tunnel (required for ingress)
minikube tunnel
```

## 📊 Monitoring

### Resource Usage
```bash
# Node resources
kubectl top nodes

# Pod resources
kubectl top pods -n oryon-ai

# HPA status
kubectl get hpa -n oryon-ai
```

### Logs
```bash
# Stream logs
kubectl logs -f deployment/fastapi -n oryon-ai

# Last 50 lines
kubectl logs --tail=50 deployment/fastapi -n oryon-ai

# All pods with label
kubectl logs -l app=celery-worker -n oryon-ai
```

## 🐛 Troubleshooting

### Pods not starting
```bash
# Check pod events
kubectl describe pod <pod-name> -n oryon-ai

# Check all events
kubectl get events -n oryon-ai --sort-by='.lastTimestamp'

# Check previous pod logs (if crashed)
kubectl logs <pod-name> -n oryon-ai --previous
```

### Image pull issues
```bash
# List images in minikube
minikube image ls | grep oryon-ai

# If missing, reload
minikube image load oryon-ai/api:latest
```

### Storage issues
```bash
# Check PVCs
kubectl get pvc -n oryon-ai

# Describe PVC for issues
kubectl describe pvc data-pvc -n oryon-ai
```

### Network issues
```bash
# Test service DNS
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup fastapi.oryon-ai.svc.cluster.local

# Test database connection
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -n oryon-ai -- \
  psql -h postgres -U postgres -d oryon
```

## 🔄 Cleanup

### Delete application only
```bash
kubectl delete namespace oryon-ai
```

### Stop cluster
```bash
minikube stop
```

### Complete cleanup
```bash
minikube delete
docker system prune -a
```

## 📚 Next Steps

1. **Deploy your app**: `./scripts/deploy-k8s.sh dev`
2. **Access dashboard**: `minikube dashboard`
3. **View logs**: `kubectl logs -f deployment/fastapi -n oryon-ai`
4. **Scale pods**: `kubectl scale deployment fastapi --replicas=3 -n oryon-ai`

## 🆘 Help

- [Minikube Docs](https://minikube.sigs.k8s.io/docs/)
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Kubernetes Docs](https://kubernetes.io/docs/home/)

---

**Your cluster is ready! Start deploying with `./scripts/deploy-k8s.sh dev`** 🚀
