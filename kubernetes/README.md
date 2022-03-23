this isnt working yet:
To use kubernetes:

minikube start

eval('minikube docker-env')

docker build -t localserve .

kubectl apply  -f kubernetes/service.yaml

kubectl apply  -f kubernetes/deployment.yaml

kubectl apply  -f kubernetes/ingress.yml    


add minikube ip to etc/host followed by yolokube.test
