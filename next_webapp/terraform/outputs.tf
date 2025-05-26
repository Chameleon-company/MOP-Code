output "gke_cluster_name" {
  value = google_container_cluster.primary.name
}

output "artifact_registry_repo_url" {
  value = "https://${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}"
}
