terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "europe-west3"
  description = "Sovereign EU region (Frankfurt)"
}

resource "google_storage_bucket" "rag_corpus" {
  name                        = "${var.project_id}-hybrid-rag-corpus"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_vertex_ai_index_endpoint" "hybrid_endpoint" {
  display_name = "vertex-hybrid-rag-endpoint"
  location     = var.region
}

resource "google_cloud_run_v2_service" "rag_api" {
  name     = "vertex-hybrid-rag-api"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/vertex-hybrid-rag:latest"
      env {
        name  = "GCP_REGION"
        value = var.region
      }
    }
  }
}
