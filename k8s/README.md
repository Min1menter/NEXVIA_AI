# Deploy on Minikube
```bash
minikube start --cpus 4 --memory 8g
eval $(minikube docker-env)                      # build images straight into Minikube
docker build --target serve     -t fraudshield-serve:latest .
docker build --target train     -t fraudshield-train:latest .
docker build --target dashboard -t fraudshield-dashboard:latest .

kubectl apply -f k8s/00-base.yaml -f k8s/10-infra.yaml
kubectl apply -f k8s/20-apps.yaml                # runs the bootstrap Job (data -> Feast -> train) + services
kubectl -n fraudshield logs -f job/bootstrap-pipeline
kubectl -n fraudshield rollout restart deploy/api deploy/consumer   # pick up models once the Job finishes

kubectl apply -f k8s/30-producer-job.yaml        # replay transactions through Kafka
minikube service -n fraudshield dashboard        # open the analyst console
```
Single-node Minikube note: the three PVCs are ReadWriteOnce, which works because every pod lands on the same node.
Change the secrets in `00-base.yaml` (and `DATABASE_URL` in `20-apps.yaml`) before sharing a cluster.
