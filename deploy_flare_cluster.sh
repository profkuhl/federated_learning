#!/bin/bash

ANSIBLE_DIR="/home/k3s-server-07/ansible"
INVENTORY="$ANSIBLE_DIR/inventory.ini"

echo "========================================"
echo "NVIDIA FLARE Cluster Deployment Script"
echo "========================================"
echo ""

if [ ! -f "$INVENTORY" ]; then
    echo "Error: Inventory file not found at $INVENTORY"
    exit 1
fi

echo "Available deployment options:"
echo "1. Deploy server only"
echo "2. Deploy clients only"
echo "3. Deploy CIFAR-10 application"
echo "4. Verify cluster"
echo "5. Configure networking"
echo "6. Full deployment (all steps)"
echo "7. Start federated learning test"
echo ""

read -p "Select option (1-7): " choice

cd "$ANSIBLE_DIR"

case $choice in
    1)
        echo "Deploying FLARE server..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_flare_server.yml
        ;;
    2)
        echo "Deploying FLARE clients..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_flare_clients.yml
        ;;
    3)
        echo "Deploying CIFAR-10 application..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_cifar10_app.yml
        ;;
    4)
        echo "Verifying cluster..."
        ansible-playbook -i "$INVENTORY" playbooks/verify_flare_cluster.yml
        ;;
    5)
        echo "Configuring networking..."
        ansible-playbook -i "$INVENTORY" playbooks/configure_networking.yml
        ;;
    6)
        echo "Full deployment starting..."
        echo "Step 1: Deploying server..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_flare_server.yml
        echo "Step 2: Deploying clients..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_flare_clients.yml
        echo "Step 3: Deploying CIFAR-10 application..."
        ansible-playbook -i "$INVENTORY" playbooks/deploy_cifar10_app.yml
        echo "Step 4: Configuring networking..."
        ansible-playbook -i "$INVENTORY" playbooks/configure_networking.yml
        echo "Step 5: Verifying cluster..."
        ansible-playbook -i "$INVENTORY" playbooks/verify_flare_cluster.yml
        echo "Full deployment completed!"
        ;;
    7)
        echo "Starting federated learning test..."
        sudo systemctl start nvflare-server
        sleep 5
        ansible -i "$INVENTORY" flare_clients -b -m systemd -a "name=nvflare-client state=started"
        echo ""
        echo "Services started. Run the test:"
        echo "/home/k3s-server-07/nvflare/test_federated_learning.sh"
        ;;
    *)
        echo "Invalid option selected."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Deployment operation completed!"
echo "========================================"