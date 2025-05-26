variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The region where resources will be deployed"
  default     = "australia-southeast1"
}

variable "zone" {
  description = "The zone for the GKE cluster"
  default     = "australia-southeast1-b"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  default     = "sushma-gke-cluster"
}

variable "artifact_repo_name" {
  description = "Name of the Artifact Registry repository"
  default     = "sushma-docker-repo"
}
variable "mongodb_uri" {
  description = "MongoDB connection string"
  type        = string
}