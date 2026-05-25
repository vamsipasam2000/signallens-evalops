# AWS ECS Fargate Deployment

This is a minimal AWS path for running SignalLens EvalOps as a FastAPI container on ECS Fargate.

## Scope

- ECR repository and image push
- ECS task definition
- ECS service on Fargate
- Environment variables
- CloudWatch logs
- Manual AWS CLI deployment commands

No Kubernetes, Terraform, CI/CD, databases, service mesh, autoscaling policy, or load balancer is included here.

## Prerequisites

- AWS CLI authenticated locally
- Docker daemon running locally
- An AWS VPC with at least one subnet
- A security group that allows inbound TCP `8000` from your IP
- ECS task execution role named `ecsTaskExecutionRole`

Create the task execution role if it does not already exist:

```bash
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://deploy/aws/ecs-task-execution-assume-role-policy.json

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

## Deployment Variables

Set these from the repository root:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export APP_NAME=signallens-evalops
export IMAGE_TAG=week1
export ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME"
export CLUSTER_NAME=signallens-evalops
export SERVICE_NAME=signallens-evalops-api
export LOG_GROUP=/ecs/signallens-evalops
```

`deploy/aws/env.example` lists the same values for quick reference. Do not put real secret values in this file.

## 1. Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name "$APP_NAME" \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256 \
  --region "$AWS_REGION"
```

If the repository already exists, continue to the push step.

## 2. Build And Push Image

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build --platform linux/amd64 -t "$APP_NAME:$IMAGE_TAG" .
docker tag "$APP_NAME:$IMAGE_TAG" "$ECR_URI:$IMAGE_TAG"
docker push "$ECR_URI:$IMAGE_TAG"
```

## 3. Create CloudWatch Log Group

```bash
aws logs create-log-group \
  --log-group-name "$LOG_GROUP" \
  --region "$AWS_REGION"

aws logs put-retention-policy \
  --log-group-name "$LOG_GROUP" \
  --retention-in-days 14 \
  --region "$AWS_REGION"
```

If the log group already exists, keep the retention command and continue.

## 4. Create ECS Cluster

```bash
aws ecs create-cluster \
  --cluster-name "$CLUSTER_NAME" \
  --region "$AWS_REGION"
```

## 5. Register Task Definition

Create a rendered task definition from `deploy/aws/task-definition.template.json` by replacing:

- `${AWS_ACCOUNT_ID}`
- `${AWS_REGION}`
- `${IMAGE_TAG}`

Then register it:

```bash
aws ecs register-task-definition \
  --cli-input-json file://deploy/aws/task-definition.rendered.json \
  --region "$AWS_REGION"
```

Capture the returned `taskDefinition.taskDefinitionArn`.

## 6. Create ECS Service

Create a rendered service file from `deploy/aws/service.template.json` by replacing:

- `${TASK_DEFINITION_ARN}`
- `subnet-REPLACE_ME`
- `sg-REPLACE_ME`

Use a public subnet and a security group that allows TCP `8000` from your IP for this lightweight setup.

```bash
aws ecs create-service \
  --cli-input-json file://deploy/aws/service.rendered.json \
  --region "$AWS_REGION"
```

## 7. Deploy A New Image

After pushing a new image tag, register a new task definition revision with that tag and update the service:

```bash
aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --task-definition "$TASK_DEFINITION_ARN" \
  --force-new-deployment \
  --region "$AWS_REGION"
```

## 8. Smoke Test

Find the running task:

```bash
aws ecs list-tasks \
  --cluster "$CLUSTER_NAME" \
  --service-name "$SERVICE_NAME" \
  --region "$AWS_REGION"
```

Describe the task, get its network interface, then fetch the public IP:

```bash
aws ecs describe-tasks \
  --cluster "$CLUSTER_NAME" \
  --tasks "$TASK_ARN" \
  --region "$AWS_REGION"

aws ec2 describe-network-interfaces \
  --network-interface-ids "$NETWORK_INTERFACE_ID" \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text \
  --region "$AWS_REGION"
```

Then call:

```bash
curl "http://$PUBLIC_IP:8000/health"
curl "http://$PUBLIC_IP:8000/v1/evals/summary"
```

## Environment Variables

The task definition sets:

| Variable | Value | Purpose |
|---|---|---|
| `APP_ENV` | `production` | Runtime environment label |
| `LOG_LEVEL` | `info` | Application log level |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Optional tracing host |

Do not place secret values directly in committed task-definition JSON. If Langfuse export is enabled later, add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` through ECS secrets or a manually managed task definition revision.

## CloudWatch Logs

Container logs are sent to:

```text
/ecs/signallens-evalops
```

Each stream uses the prefix:

```text
api/signallens-api/<task-id>
```

## Notes

This deployment intentionally uses a public task IP and no load balancer to stay small. That is fine for a portfolio smoke test. A stable production endpoint would normally add an Application Load Balancer, HTTPS, autoscaling, and stricter network boundaries, which are intentionally out of scope for this step.
