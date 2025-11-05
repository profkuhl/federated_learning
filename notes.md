UFW firewall needs to open ports 8002, 8003
disallow previous network, allow networkd
    ```bash
    sudo systemctl status NetworkManager
    sudo systemctl stop NetworkManager
    sudo systemctl disable NetworkManager
    sudo systemctl enable systemd-networkd
    sudo systemctl start systemd-networkd
    netplan apply
    sudo netplan apply
    ```

provision -> ansible deploy_docker_nvflare, restart local docker container


```bash
● Bash(docker stop flserver 2>/dev/null; ansible nvflare_clients -i /home/k3s-server-07/federated_learning/ansible/inventory.ini -m shell -a "docker stop {{ inventory_…)      
  ⎿  flserver                                                                                                                                                                  
     k3s-client-09 | CHANGED | rc=0 >>                                                                                                                                         
     k3s-client-09                                                                                                                                                             
     … +2 lines (ctrl+o to expand)                                                                                                                                             
                                                                                                                                                                               
● Now re-provision and deploy (this will take a few minutes):                                                                                                                  
                                           
  cd /home/k3s-server-07/federated_learning                                                                                                                                    
  ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml 

  After it completes, start the containers: 

  # Start server
  cd /home/k3s-server-07/nvflare_workspace/server/startup && ./docker.sh -d

  # Start clients
  ansible nvflare_clients -i ansible/inventory.ini -m shell -a "cd /home/{{ ansible_user }}/nvflare_workspace/{{ inventory_hostname }}/startup && ./docker.sh -d"

  Then try the admin console again:

  cd /home/k3s-server-07/federated_learning
  ./admin_docker.sh[]
```