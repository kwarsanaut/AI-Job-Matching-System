#!/bin/bash

# ======================================
# AI Job Matching System - Quick Setup Script
# This script sets up the complete infrastructure
# ======================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="AI Job Matching System"
DOCKER_COMPOSE_VERSION="3.8"
MIN_DOCKER_VERSION="20.10"
MIN_DOCKER_COMPOSE_VERSION="2.0"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "======================================"
    echo "$1"
    echo "======================================"
    echo -e "${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker version
check_docker_version() {
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    print_status "Docker version: $DOCKER_VERSION"
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_status "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    COMPOSE_VERSION=$(docker-compose --version | grep -oP '\d+\.\d+' | head -1)
    print_status "Docker Compose version: $COMPOSE_VERSION"
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data/postgres"
        "data/redis" 
        "data/qdrant"
        "data/prometheus"
        "data/grafana"
        "data/elasticsearch"
        "data/pgadmin"
        "logs/api"
        "logs/worker"
        "logs/ai_service"
        "logs/nginx"
        "uploads"
        "backups"
        "ai_models"
        "database/migrations"
        "nginx/conf.d"
        "nginx/ssl"
        "monitoring/prometheus"
        "monitoring/grafana/provisioning/datasources"
        "monitoring/grafana/provisioning/dashboards"
        "monitoring/grafana/dashboards"
        "monitoring/pgadmin"
        "redis"
        "qdrant"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "Directory structure created successfully"
}

# Function to setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env file from template"
        else
            print_warning ".env.example not found, creating basic .env file"
            cat > .env << EOF
# Basic configuration
ENVIRONMENT=development
DB_PASSWORD=secure_db_password_123
REDIS_PASSWORD=redis_password_123
JWT_SECRET=$(openssl rand -base64 32)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
EOF
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Check for required API keys
    if grep -q "your_openai_api_key_here" .env; then
        print_warning "Please update OPENAI_API_KEY in .env file for full AI functionality"
    fi
    
    if grep -q "your_huggingface_api_key_here" .env; then
        print_warning "Please update HUGGINGFACE_API_KEY in .env file for additional models"
    fi
}

# Function to create basic configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Nginx configuration
    if [ ! -f "nginx/nginx.conf" ]; then
        cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8080;
    }
    
    upstream ai_backend {
        server ai_service:8000;
    }
    
    upstream frontend_backend {
        server frontend:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ai:10m rate=2r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Frontend
        location / {
            proxy_pass http://frontend_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # AI service routes
        location /ai/ {
            limit_req zone=ai burst=5 nodelay;
            proxy_pass http://ai_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks
        location /health {
            access_log off;
            proxy_pass http://api_backend;
        }
    }
}
EOF
        print_success "Created nginx configuration"
    fi
    
    # Prometheus configuration
    if [ ! -f "monitoring/prometheus/prometheus.yml" ]; then
        cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'job-matching-api'
    static_configs:
      - targets: ['api:8080']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'job-matching-ai'
    static_configs:
      - targets: ['ai_service:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
EOF
        print_success "Created Prometheus configuration"
    fi
    
    # Grafana datasource
    if [ ! -f "monitoring/grafana/provisioning/datasources/prometheus.yml" ]; then
        cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
        print_success "Created Grafana datasource configuration"
    fi
    
    # Redis configuration
    if [ ! -f "redis/redis.conf" ]; then
        cat > redis/redis.conf << 'EOF'
# Redis configuration for job matching system
port 6379
bind 0.0.0.0
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./
maxmemory 512mb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
EOF
        print_success "Created Redis configuration"
    fi
    
    # Qdrant configuration
    if [ ! -f "qdrant/config.yaml" ]; then
        cat > qdrant/config.yaml << 'EOF'
service:
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: ./storage
  snapshots_path: ./snapshots
  on_disk_payload: true
  wal:
    wal_capacity_mb: 32
    wal_segments_ahead: 0

cluster:
  enabled: false

telemetry_disabled: true
EOF
        print_success "Created Qdrant configuration"
    fi
}

# Function to create database initialization script
create_database_init() {
    print_status "Creating database initialization script..."
    
    if [ ! -f "database/init.sql" ]; then
        cat > database/init.sql << 'EOF'
-- Database initialization script
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create application user
CREATE USER job_matching_app WITH PASSWORD 'secure_app_password';
GRANT CONNECT ON DATABASE job_matching_ai TO job_matching_app;
GRANT USAGE ON SCHEMA public TO job_matching_app;
GRANT CREATE ON SCHEMA public TO job_matching_app;

-- Set timezone
SET timezone = 'UTC';

-- Configure shared_preload_libraries for pg_stat_statements
-- Note: This requires restart in production
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Create extension for query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT 'Database initialized successfully' as status;
EOF
        print_success "Created database initialization script"
    fi
}

# Function to create sample data script
create_sample_data() {
    print_status "Creating sample data script..."
    
    if [ ! -f "database/seed.sql" ]; then
        cat > database/seed.sql << 'EOF'
-- Sample data for development and testing

-- Insert additional sample companies
INSERT INTO companies (name, description, location, size, industry, website) VALUES
('StartupTech', 'Innovative startup focused on AI solutions', 'Amsterdam, Netherlands', 'startup', 'Technology', 'https://startuptech.nl'),
('BigCorp International', 'Large multinational corporation', 'Frankfurt, Germany', 'enterprise', 'Finance', 'https://bigcorp.com'),
('Creative Agency', 'Digital marketing and design agency', 'Barcelona, Spain', 'small', 'Marketing', 'https://creative-agency.es'),
('DataScience Pro', 'Data analytics consulting firm', 'Milan, Italy', 'medium', 'Data Science', 'https://datascience-pro.it'),
('CloudSolutions Ltd', 'Cloud infrastructure specialists', 'London, UK', 'medium', 'Cloud Computing', 'https://cloudsolutions.uk')
ON CONFLICT (name) DO NOTHING;

-- Insert sample users for testing
INSERT INTO users (email, password_hash, role, is_active, email_verified) VALUES
('admin@jobmatch.ai', crypt('admin123', gen_salt('bf')), 'admin', true, true),
('recruiter@techcorp.eu', crypt('recruiter123', gen_salt('bf')), 'recruiter', true, true),
('candidate@example.com', crypt('candidate123', gen_salt('bf')), 'candidate', true, true)
ON CONFLICT (email) DO NOTHING;

-- Update profile completion for existing candidates
UPDATE candidates SET profile_completion = calculate_profile_completion(id);

-- Insert some initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, tags) VALUES
('system_startup', 1, '{"event": "initial_setup", "version": "1.0.0"}'),
('sample_data_loaded', 1, '{"timestamp": "' || NOW() || '"}');

-- Create initial system configuration if not exists
INSERT INTO system_config (key, value, description, data_type, is_public) VALUES
('system_initialized', 'true', 'System has been initialized', 'boolean', false),
('sample_data_version', '1.0.0', 'Version of sample data loaded', 'string', false),
('last_maintenance', NOW()::text, 'Last maintenance timestamp', 'string', false)
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = NOW();

SELECT 'Sample data loaded successfully' as status;
EOF
        print_success "Created sample data script"
    fi
}

# Function to check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check Docker
    check_docker_version
    
    # Check available disk space (minimum 5GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=5242880  # 5GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        print_warning "Available disk space: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
        print_warning "Recommended minimum: 5GB"
    else
        print_success "Sufficient disk space available"
    fi
    
    # Check available memory
    if command_exists free; then
        AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$AVAILABLE_MEMORY" -lt "4096" ]; then
            print_warning "Available memory: ${AVAILABLE_MEMORY}MB"
            print_warning "Recommended minimum: 4GB"
        else
            print_success "Sufficient memory available"
        fi
    fi
}

# Function to validate configuration
validate_config() {
    print_status "Validating configuration..."
    
    # Check if .env file exists and has required variables
    if [ ! -f ".env" ]; then
        print_error ".env file not found"
        return 1
    fi
    
    # Check for required environment variables
    required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env; then
            print_error "Required environment variable $var not found in .env"
            return 1
        fi
    done
    
    print_success "Configuration validation passed"
}

# Function to pull Docker images
pull_images() {
    print_header "Pulling Docker Images"
    
    images=(
        "pgvector/pgvector:pg15"
        "redis:7-alpine"
        "qdrant/qdrant:v1.7.0"
        "nginx:alpine"
        "prom/prometheus:v2.45.0"
        "grafana/grafana:10.0.0"
        "jaegertracing/all-in-one:1.47"
    )
    
    for image in "${images[@]}"; do
        print_status "Pulling $image..."
        if docker pull "$image"; then
            print_success "Successfully pulled $image"
        else
            print_error "Failed to pull $image"
            exit 1
        fi
    done
}

# Function to start services
start_services() {
    print_header "Starting Services"
    
    # Start core services first
    print_status "Starting database services..."
    docker-compose up -d postgres redis qdrant
    
    # Wait for databases to be ready
    print_status "Waiting for databases to be ready..."
    sleep 30
    
    # Check database health
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U jobmatcher -d job_matching_ai; then
            print_success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "PostgreSQL failed to start"
            exit 1
        fi
        sleep 2
    done
    
    # Start application services
    print_status "Starting application services..."
    docker-compose up -d api ai_service worker
    
    # Wait for application services
    sleep 20
    
    # Start monitoring services
    print_status "Starting monitoring services..."
    docker-compose up -d prometheus grafana jaeger nginx
    
    # Start development tools if in development mode
    if grep -q "ENVIRONMENT=development" .env; then
        print_status "Starting development tools..."
        docker-compose --profile development up -d
    fi
}

# Function to verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    services=(
        "postgres:5432"
        "redis:6379"
        "qdrant:6333"
        "api:8080"
        "ai_service:8000"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo "$service" | cut -d':' -f1)
        port=$(echo "$service" | cut -d':' -f2)
        
        print_status "Checking $service_name on port $port..."
        
        for i in {1..10}; do
            if docker-compose exec -T "$service_name" nc -z localhost "$port" 2>/dev/null; then
                print_success "$service_name is running"
                break
            fi
            if [ $i -eq 10 ]; then
                print_warning "$service_name may not be running properly"
            fi
            sleep 3
        done
    done
    
    # Test API health endpoint
    print_status "Testing API health endpoint..."
    sleep 10
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_warning "API health check failed - service may still be starting"
    fi
    
    # Test AI service health endpoint
    print_status "Testing AI service health endpoint..."
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "AI service health check passed"
    else
        print_warning "AI service health check failed - service may still be starting"
    fi
}

# Function to display service URLs
show_service_urls() {
    print_header "Service URLs"
    
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Available Services:${NC}"
    echo ""
    echo -e "${YELLOW}Core Services:${NC}"
    echo "  🌐 Frontend:           http://localhost:3000"
    echo "  🔌 API:                http://localhost:8080"
    echo "  🤖 AI Service:         http://localhost:8000"
    echo "  📊 API Documentation:  http://localhost:8080/docs"
    echo ""
    echo -e "${YELLOW}Monitoring:${NC}"
    echo "  📈 Grafana:            http://localhost:3001 (admin/admin123)"
    echo "  📊 Prometheus:         http://localhost:9090"
    echo "  🔍 Jaeger:             http://localhost:16686"
    echo ""
    echo -e "${YELLOW}Development Tools:${NC}"
    echo "  🗄️  pgAdmin:            http://localhost:5050"
    echo "  💾 Redis Commander:    http://localhost:8081"
    echo "  📧 MailHog:            http://localhost:8025"
    echo ""
    echo -e "${YELLOW}Database Connections:${NC}"
    echo "  🐘 PostgreSQL:         localhost:5432"
    echo "  🔴 Redis:              localhost:6379"
    echo "  🔍 Qdrant:             localhost:6333"
    echo ""
    echo -e "${BLUE}📖 Next Steps:${NC}"
    echo "  1. Visit http://localhost:3000 to access the frontend"
    echo "  2. Check API documentation at http://localhost:8080/docs"
    echo "  3. Monitor services at http://localhost:3001"
    echo "  4. Update .env file with your API keys for full functionality"
    echo ""
    echo -e "${GREEN}🚀 Happy job matching!${NC}"
}

# Function to show logs
show_logs() {
    print_header "Service Logs"
    echo "To view logs for specific services:"
    echo ""
    echo "  docker-compose logs -f api          # API logs"
    echo "  docker-compose logs -f ai_service   # AI service logs"
    echo "  docker-compose logs -f worker       # Worker logs"
    echo "  docker-compose logs -f postgres     # Database logs"
    echo "  docker-compose logs                 # All service logs"
    echo ""
}

# Function to clean up on failure
cleanup_on_failure() {
    print_error "Setup failed. Cleaning up..."
    docker-compose down --remove-orphans
    print_status "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --dev               Setup development environment with all tools"
    echo "  --prod              Setup production environment"
    echo "  --minimal           Setup minimal environment (core services only)"
    echo "  --pull-images       Only pull Docker images"
    echo "  --check-only        Only check requirements"
    echo "  --logs              Show log viewing commands"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full setup with default options"
    echo "  $0 --dev            # Development setup with all tools"
    echo "  $0 --minimal        # Minimal setup for testing"
    echo "  $0 --check-only     # Just check system requirements"
    echo ""
}

# Main setup function
main() {
    print_header "$PROJECT_NAME Setup"
    
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --logs)
            show_logs
            exit 0
            ;;
        --check-only)
            check_requirements
            exit 0
            ;;
        --pull-images)
            pull_images
            exit 0
            ;;
        --dev)
            echo "ENVIRONMENT=development" > .env.override
            ;;
        --prod)
            echo "ENVIRONMENT=production" > .env.override
            ;;
        --minimal)
            echo "MINIMAL_SETUP=true" > .env.override
            ;;
    esac
    
    # Trap to cleanup on failure
    trap cleanup_on_failure ERR
    
    # Main setup steps
    print_status "Starting setup process..."
    
    check_requirements
    create_directories
    setup_environment
    create_config_files
    create_database_init
    create_sample_data
    validate_config
    pull_images
    start_services
    verify_deployment
    show_service_urls
    
    print_success "Setup completed successfully!"
    
    # Remove trap
    trap - ERR
}

# Run main function with all arguments
main "$@"
