#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Azure Container Apps — Deploy Archive Classifier
#
# Prerequisites:
#   1. Azure CLI installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. Logged in: az login
#   3. Copy deploy/.env.aca.example → deploy/.env.aca and fill in values
#
# Usage:
#   cd "Corporate Archives/deploy"
#   chmod +x azure-container-apps.sh
#   ./azure-container-apps.sh
#
# Cost: ~£3-8/month with scale-to-zero (pay only when accessed)
# ============================================================

# ── Load config ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env.aca" ]]; then
    source "$SCRIPT_DIR/.env.aca"
else
    echo "❌ Missing deploy/.env.aca — copy from .env.aca.example and fill in values"
    exit 1
fi

# ── Defaults (override in .env.aca) ──
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-archive-classifier}"
LOCATION="${LOCATION:-uksouth}"
ACR_NAME="${ACR_NAME:-acrarchivedocs}"
ENV_NAME="${ENV_NAME:-aca-env-archive}"
APP_NAME="${APP_NAME:-archive-doc-classifier}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# These MUST be set in .env.aca
: "${DATABRICKS_HOST:?Set DATABRICKS_HOST in .env.aca}"
: "${DATABRICKS_TOKEN:?Set DATABRICKS_TOKEN in .env.aca}"
: "${TRAINING_JOB_ID:?Set TRAINING_JOB_ID in .env.aca}"

MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-/Users/hakeemfujah@tfl.gov.uk/experiments/ac_model_v2}"

APP_DIR="$SCRIPT_DIR/../Autoclassification Scheme/streamlit_demo/streamlit_app"

echo "============================================================"
echo "  Azure Container Apps — Archive Classifier Deployment"
echo "============================================================"
echo "  Resource Group:  $RESOURCE_GROUP"
echo "  Location:        $LOCATION"
echo "  ACR:             $ACR_NAME"
echo "  App Name:        $APP_NAME"
echo "  Dockerfile:      $APP_DIR/Dockerfile"
echo "============================================================"
echo ""

# ── 1. Resource Group ──
echo "── 1. Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
echo "   ✅ $RESOURCE_GROUP in $LOCATION"

# ── 2. Azure Container Registry ──
echo ""
echo "── 2. Creating container registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none
echo "   ✅ $ACR_NAME.azurecr.io"

# ── 3. Build & push image ──
echo ""
echo "── 3. Building and pushing Docker image (ACR cloud build)..."
az acr build \
    --registry "$ACR_NAME" \
    --image "${APP_NAME}:${IMAGE_TAG}" \
    --file "$APP_DIR/Dockerfile" \
    "$APP_DIR"
echo "   ✅ Image: ${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}"

# ── 4. Container Apps Environment ──
echo ""
echo "── 4. Creating Container Apps environment..."
az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
echo "   ✅ Environment: $ENV_NAME"

# ── 5. Get ACR credentials ──
ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# ── 6. Deploy Container App ──
echo ""
echo "── 5. Deploying container app (scale-to-zero)..."
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "${ACR_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 2 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --secrets \
        "databricks-host=$DATABRICKS_HOST" \
        "databricks-token=$DATABRICKS_TOKEN" \
        "training-job-id=$TRAINING_JOB_ID" \
    --env-vars \
        "DATABRICKS_HOST=secretref:databricks-host" \
        "DATABRICKS_TOKEN=secretref:databricks-token" \
        "MLFLOW_TRACKING_URI=databricks" \
        "MLFLOW_EXPERIMENT=$MLFLOW_EXPERIMENT" \
        "TRAINING_JOB_ID=secretref:training-job-id" \
    --output none

# ── 7. Get the URL ──
APP_URL=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" \
    -o tsv)

echo ""
echo "============================================================"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "============================================================"
echo "  App URL:  https://${APP_URL}"
echo "  Console:  https://portal.azure.com/#@/resource/subscriptions/"
echo ""
echo "  Scale:    0-2 replicas (scale-to-zero when idle)"
echo "  CPU:      0.5 vCPU / 1 GiB RAM"
echo "  Cost:     ~£3-8/month at low traffic (£0 when idle)"
echo ""
echo "  Share this link with anyone — no Databricks login needed."
echo "============================================================"
echo ""
echo "  Useful commands:"
echo "    # View logs:"
echo "    az containerapp logs show -n $APP_NAME -g $RESOURCE_GROUP --follow"
echo ""
echo "    # Update image after code changes:"
echo "    az acr build --registry $ACR_NAME --image ${APP_NAME}:latest --file \"$APP_DIR/Dockerfile\" \"$APP_DIR\""
echo "    az containerapp update -n $APP_NAME -g $RESOURCE_GROUP --image ${ACR_SERVER}/${APP_NAME}:latest"
echo ""
echo "    # Tear down (stop billing):"
echo "    az group delete -n $RESOURCE_GROUP --yes --no-wait"
echo "============================================================"