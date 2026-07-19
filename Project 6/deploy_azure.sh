#!/bin/bash
# ============================================================================
# Reference deployment script for Project 3 -> Azure
# Run these in order, section by section (don't blindly run the whole file --
# read each section, since a couple of names must be globally unique to you).
# ============================================================================
set -e

# ---- Variables: EDIT THESE ----
RESOURCE_GROUP="project3-rg"
LOCATION="eastus"
STORAGE_ACCOUNT="project3storage$RANDOM"     # must be globally unique, lowercase, no dashes
CONTAINER_NAME="model-artifacts"
ACR_NAME="project3acr$RANDOM"                # must be globally unique
IMAGE_NAME="wine-pairing-api"
ACI_NAME="wine-pairing-api-aci"
VNET_NAME="project3-vnet"
SUBNET_NAME="project3-subnet"
NSG_NAME="project3-nsg"
DNS_LABEL="project3wine$RANDOM"              # must be globally unique

# ---- 0. Login and resource group ----
az login
az group create --name $RESOURCE_GROUP --location $LOCATION

# ---- 1. Persistent storage: Blob Storage ----
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

az storage container create \
  --account-name $STORAGE_ACCOUNT \
  --name $CONTAINER_NAME \
  --auth-mode login

az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --container-name $CONTAINER_NAME \
  --name model_artifacts.joblib \
  --file model_artifacts.joblib \
  --auth-mode login

az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --container-name $CONTAINER_NAME \
  --name wine_food_pairings.csv \
  --file wine_food_pairings.csv \
  --auth-mode login

# Grab the connection string -- you'll pass this to the container as an env var
STORAGE_CONN_STR=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)
echo "Storage connection string (save this): $STORAGE_CONN_STR"

# ---- 2. Compute: Azure Container Registry + build/push image ----
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Build the image directly in ACR (no local Docker needed)
az acr build \
  --registry $ACR_NAME \
  --image $IMAGE_NAME:v1 \
  .

ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

# ---- 3. Network security: VNet + Subnet + NSG ----
az network vnet create \
  --resource-group $RESOURCE_GROUP \
  --name $VNET_NAME \
  --address-prefix 10.0.0.0/16 \
  --subnet-name $SUBNET_NAME \
  --subnet-prefix 10.0.0.0/24

az network nsg create \
  --resource-group $RESOURCE_GROUP \
  --name $NSG_NAME

# Allow inbound HTTPS/API traffic only from your own IP (replace with your real IP)
MY_IP=$(curl -s ifconfig.me)
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name $NSG_NAME \
  --name AllowMyIP-API \
  --priority 100 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes "$MY_IP/32" \
  --destination-port-ranges 8000 \
  --destination-address-prefixes '*'

# Deny all other inbound traffic
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name $NSG_NAME \
  --name DenyAllInbound \
  --priority 4096 \
  --direction Inbound \
  --access Deny \
  --protocol '*' \
  --source-address-prefixes '*' \
  --destination-port-ranges '*' \
  --destination-address-prefixes '*'

# Attach NSG to the subnet
az network vnet subnet update \
  --resource-group $RESOURCE_GROUP \
  --vnet-name $VNET_NAME \
  --name $SUBNET_NAME \
  --network-security-group $NSG_NAME

# ---- 4. Deploy the container into that VNet/subnet ----
az container create \
  --resource-group $RESOURCE_GROUP \
  --name $ACI_NAME \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:v1" \
  --registry-login-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --vnet $VNET_NAME \
  --subnet $SUBNET_NAME \
  --ports 8000 \
  --environment-variables MODEL_SOURCE=blob BLOB_CONTAINER=$CONTAINER_NAME BLOB_NAME=model_artifacts.joblib \
  --secure-environment-variables AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN_STR" \
  --cpu 1 \
  --memory 1.5

# Note: containers deployed into a VNet don't get a public IP/FQDN directly --
# that's the point (private by default). To test from your machine, either:
#   (a) use `az container exec` to shell into it and curl localhost, or
#   (b) put an Azure Application Gateway / Bastion host in front for controlled access, or
#   (c) for a simpler class demo, deploy WITHOUT --vnet/--subnet first to get a public
#       DNS label to test end-to-end, then redo with VNet attached for the security requirement.

echo "Deployment steps complete. Check resources with: az resource list --resource-group $RESOURCE_GROUP -o table"
