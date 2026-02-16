# Kubernetes Configuration Guide

## Required Configuration Before Deployment

### 1. Domain Configuration

Update the domain name in your environment-specific overlay:

**For Production (`k8s/overlays/prod/kustomization.yaml`):**
```yaml
configMapGenerator:
  - name: cloud-config
    literals:
      - DOMAIN_NAME=oryon-ai.com  # Replace with your actual domain
```

This will create:
- `oryon-ai.com` (main frontend)
- `api.oryon-ai.com` (API backend)

**DNS Requirements:**
- Create A/AAAA records pointing to your ingress controller's external IP
- Example:
  ```
  oryon-ai.com       A    203.0.113.10
  api.oryon-ai.com   A    203.0.113.10
  ```

### 2. AWS Account Configuration

If using AWS EKS with External Secrets Operator:

```yaml
configMapGenerator:
  - name: cloud-config
    literals:
      - AWS_ACCOUNT_ID=123456789012  # Replace with your 12-digit AWS account ID
```

### 3. TLS/SSL Certificate

The ingress uses cert-manager with Let's Encrypt. Ensure:

1. **cert-manager is installed:**
   ```bash
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   ```

2. **ClusterIssuer exists:**
   ```yaml
   apiVersion: cert-manager.io/v1
   kind: ClusterIssuer
   metadata:
     name: letsencrypt-prod
   spec:
     acme:
       server: https://acme-v02.api.letsencrypt.org/directory
       email: admin@oryon-ai.com  # Replace with your email
       privateKeySecretRef:
         name: letsencrypt-prod
       solvers:
       - http01:
           ingress:
             class: nginx
   ```

### 4. Verification Steps

Before deploying:

1. **Verify Kustomize output:**
   ```bash
   kubectl kustomize k8s/overlays/prod
   ```

2. **Check domain substitution:**
   ```bash
   kubectl kustomize k8s/overlays/prod | grep -A 5 "kind: Ingress"
   ```

3. **Validate DNS resolution:**
   ```bash
   nslookup api.oryon-ai.com
   nslookup oryon-ai.com
   ```

4. **Test ingress controller:**
   ```bash
   kubectl get ingressclass
   kubectl get pods -n ingress-nginx
   ```

### 5. Deploy

```bash
# Deploy to production
kubectl apply -k k8s/overlays/prod

# Verify ingress
kubectl get ingress -n oryon-ai-prod

# Check certificate issuance
kubectl get certificate -n oryon-ai-prod
kubectl describe certificate oryon-tls -n oryon-ai-prod
```

### 6. Troubleshooting

**Certificate not issuing:**
```bash
kubectl describe certificaterequest -n oryon-ai-prod
kubectl logs -n cert-manager deployment/cert-manager
```

**Ingress not working:**
```bash
kubectl describe ingress oryon-ingress -n oryon-ai-prod
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

**Domain not resolving:**
- Verify DNS records are propagated: `dig +short api.oryon-ai.com`
- Check ingress external IP: `kubectl get svc -n ingress-nginx`

## Environment-Specific Overlays

- **Dev:** `k8s/overlays/dev/` - Uses dev domain (e.g., `dev.oryon-ai.com`)
- **Staging:** `k8s/overlays/staging/` - Uses staging domain (e.g., `staging.oryon-ai.com`)
- **Prod:** `k8s/overlays/prod/` - Uses production domain (e.g., `oryon-ai.com`)

Each overlay should have its own `DOMAIN_NAME` configured in the kustomization.yaml.
