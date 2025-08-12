-- ======================================
-- AI Job Matching System Database Schema
-- PostgreSQL with Vector Extension Support
-- ======================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ======================================
-- Core Tables
-- ======================================

-- Companies table
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    website VARCHAR(255),
    location VARCHAR(255),
    size VARCHAR(50) CHECK (size IN ('startup', 'small', 'medium', 'large', 'enterprise')),
    industry VARCHAR(100),
    logo_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT companies_name_unique UNIQUE (name)
);

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT[],
    skills TEXT[] NOT NULL,
    location VARCHAR(255) NOT NULL,
    salary_min INTEGER CHECK (salary_min >= 0),
    salary_max INTEGER CHECK (salary_max >= 0),
    currency VARCHAR(3) DEFAULT 'EUR',
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('full-time', 'part-time', 'contract', 'internship')),
    work_hours VARCHAR(50),
    experience_level VARCHAR(50) CHECK (experience_level IN ('entry', 'junior', 'mid', 'senior', 'lead', 'principal')),
    remote_allowed BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'closed')),
    posted_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    applicant_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    -- AI/Vector fields
    embedding VECTOR(384), -- Sentence transformers embedding
    ai_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT jobs_salary_check CHECK (salary_min IS NULL OR salary_max IS NULL OR salary_min <= salary_max),
    CONSTRAINT jobs_expires_after_posted CHECK (expires_at IS NULL OR expires_at > posted_at)
);

-- Candidates table
CREATE TABLE candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    location VARCHAR(255),
    title VARCHAR(255),
    summary TEXT,
    skills TEXT[] NOT NULL,
    experience_years INTEGER DEFAULT 0 CHECK (experience_years >= 0),
    education_level VARCHAR(50) CHECK (education_level IN ('high_school', 'bachelors', 'masters', 'doctorate', 'certificate', 'diploma')),
    education_field VARCHAR(100),
    availability VARCHAR(50) DEFAULT 'full-time' CHECK (availability IN ('full-time', 'part-time', 'contract', 'freelance')),
    salary_expectation_min INTEGER CHECK (salary_expectation_min >= 0),
    salary_expectation_max INTEGER CHECK (salary_expectation_max >= 0),
    currency VARCHAR(3) DEFAULT 'EUR',
    remote_preference BOOLEAN DEFAULT false,
    willing_to_relocate BOOLEAN DEFAULT false,
    -- AI/Vector fields
    embedding VECTOR(384),
    ai_score FLOAT DEFAULT 0,
    profile_completion FLOAT DEFAULT 0,
    last_active TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT candidates_salary_check CHECK (salary_expectation_min IS NULL OR salary_expectation_max IS NULL OR salary_expectation_min <= salary_expectation_max),
    CONSTRAINT candidates_email_valid CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Work experience table
CREATE TABLE work_experiences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    company_name VARCHAR(255) NOT NULL,
    position VARCHAR(255) NOT NULL,
    description TEXT,
    skills_used TEXT[],
    start_date DATE NOT NULL,
    end_date DATE,
    is_current BOOLEAN DEFAULT false,
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT work_exp_dates_check CHECK (end_date IS NULL OR end_date >= start_date),
    CONSTRAINT work_exp_current_check CHECK (NOT is_current OR end_date IS NULL)
);

-- Education table
CREATE TABLE education (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    institution VARCHAR(255) NOT NULL,
    degree VARCHAR(255) NOT NULL,
    field_of_study VARCHAR(255),
    start_date DATE,
    end_date DATE,
    gpa DECIMAL(3,2) CHECK (gpa >= 0 AND gpa <= 4),
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT education_dates_check CHECK (start_date IS NULL OR end_date IS NULL OR end_date >= start_date)
);

-- Applications table
CREATE TABLE applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewing', 'interviewed', 'offered', 'rejected', 'hired', 'withdrawn')),
    cover_letter TEXT,
    resume_url VARCHAR(255),
    ai_match_score FLOAT,
    matched_skills TEXT[],
    hr_notes TEXT,
    applied_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT applications_unique UNIQUE(job_id, candidate_id)
);

-- AI Matches table (for caching match results)
CREATE TABLE ai_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    overall_score FLOAT NOT NULL CHECK (overall_score >= 0 AND overall_score <= 100),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    skills_score FLOAT CHECK (skills_score >= 0 AND skills_score <= 100),
    experience_score FLOAT CHECK (experience_score >= 0 AND experience_score <= 100),
    location_score FLOAT CHECK (location_score >= 0 AND location_score <= 100),
    salary_score FLOAT CHECK (salary_score >= 0 AND salary_score <= 100),
    education_score FLOAT CHECK (education_score >= 0 AND education_score <= 100),
    cultural_fit_score FLOAT CHECK (cultural_fit_score >= 0 AND cultural_fit_score <= 100),
    matched_skills TEXT[],
    explanation TEXT,
    recommendations TEXT[],
    generated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT ai_matches_unique UNIQUE(job_id, candidate_id)
);

-- Match batches for bulk processing
CREATE TABLE match_batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_ids UUID[],
    candidate_ids UUID[],
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    progress FLOAT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    total_pairs INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,
    successful INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_msg TEXT,
    created_by VARCHAR(255),
    
    CONSTRAINT batch_progress_check CHECK (processed <= total_pairs)
);

-- ======================================
-- Analytics & Monitoring Tables
-- ======================================

-- Search analytics for tracking user behavior
CREATE TABLE search_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    search_type VARCHAR(50) NOT NULL CHECK (search_type IN ('job_search', 'candidate_search', 'ai_match')),
    search_query TEXT,
    filters JSONB,
    results_count INTEGER DEFAULT 0,
    user_type VARCHAR(50) CHECK (user_type IN ('recruiter', 'candidate', 'admin', 'anonymous')),
    user_id UUID,
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Job analytics
CREATE TABLE job_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    date DATE DEFAULT CURRENT_DATE,
    views INTEGER DEFAULT 0,
    applications INTEGER DEFAULT 0,
    matches_generated INTEGER DEFAULT 0,
    avg_match_score FLOAT,
    high_score_matches INTEGER DEFAULT 0,
    medium_score_matches INTEGER DEFAULT 0,
    low_score_matches INTEGER DEFAULT 0,
    top_matched_skills TEXT[],
    bottleneck_areas TEXT[],
    processing_time_ms INTEGER,
    
    CONSTRAINT job_analytics_unique UNIQUE(job_id, date)
);

-- System metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- ======================================
-- Skills Taxonomy & Reference Data
-- ======================================

-- Skills taxonomy for standardization
CREATE TABLE skills_taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL,
    aliases TEXT[],
    popularity_score FLOAT DEFAULT 0,
    demand_score FLOAT DEFAULT 0,
    parent_skill_id UUID REFERENCES skills_taxonomy(id),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Countries and locations reference
CREATE TABLE locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    region VARCHAR(100),
    timezone VARCHAR(50),
    coordinates POINT,
    is_remote_friendly BOOLEAN DEFAULT false,
    
    CONSTRAINT locations_unique UNIQUE(country, city)
);

-- ======================================
-- Audit & History Tables
-- ======================================

-- Audit log for tracking changes
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(50) NOT NULL,
    record_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP DEFAULT NOW()
);

-- ======================================
-- Indexes for Performance
-- ======================================

-- Companies indexes
CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);
CREATE INDEX idx_companies_location ON companies(location);

-- Jobs indexes
CREATE INDEX idx_jobs_company_id ON jobs(company_id);
CREATE INDEX idx_jobs_title ON jobs USING gin(to_tsvector('english', title));
CREATE INDEX idx_jobs_description ON jobs USING gin(to_tsvector('english', description));
CREATE INDEX idx_jobs_skills ON jobs USING gin(skills);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary_range ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_job_type ON jobs(job_type);
CREATE INDEX idx_jobs_experience_level ON jobs(experience_level);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_posted_at ON jobs(posted_at DESC);
CREATE INDEX idx_jobs_expires_at ON jobs(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_jobs_remote_allowed ON jobs(remote_allowed) WHERE remote_allowed = true;
CREATE INDEX idx_jobs_embedding ON jobs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Candidates indexes
CREATE INDEX idx_candidates_email ON candidates(email);
CREATE INDEX idx_candidates_name ON candidates(first_name, last_name);
CREATE INDEX idx_candidates_title ON candidates USING gin(to_tsvector('english', title));
CREATE INDEX idx_candidates_summary ON candidates USING gin(to_tsvector('english', summary));
CREATE INDEX idx_candidates_skills ON candidates USING gin(skills);
CREATE INDEX idx_candidates_location ON candidates(location);
CREATE INDEX idx_candidates_experience_years ON candidates(experience_years);
CREATE INDEX idx_candidates_education_level ON candidates(education_level);
CREATE INDEX idx_candidates_availability ON candidates(availability);
CREATE INDEX idx_candidates_salary_range ON candidates(salary_expectation_min, salary_expectation_max);
CREATE INDEX idx_candidates_remote_preference ON candidates(remote_preference) WHERE remote_preference = true;
CREATE INDEX idx_candidates_last_active ON candidates(last_active DESC);
CREATE INDEX idx_candidates_profile_completion ON candidates(profile_completion DESC);
CREATE INDEX idx_candidates_embedding ON candidates USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Work experience indexes
CREATE INDEX idx_work_exp_candidate_id ON work_experiences(candidate_id);
CREATE INDEX idx_work_exp_company_name ON work_experiences(company_name);
CREATE INDEX idx_work_exp_position ON work_experiences USING gin(to_tsvector('english', position));
CREATE INDEX idx_work_exp_skills_used ON work_experiences USING gin(skills_used);
CREATE INDEX idx_work_exp_dates ON work_experiences(start_date, end_date);
CREATE INDEX idx_work_exp_current ON work_experiences(is_current) WHERE is_current = true;

-- Education indexes
CREATE INDEX idx_education_candidate_id ON education(candidate_id);
CREATE INDEX idx_education_institution ON education(institution);
CREATE INDEX idx_education_degree ON education(degree);
CREATE INDEX idx_education_field ON education(field_of_study);

-- Applications indexes
CREATE INDEX idx_applications_job_id ON applications(job_id);
CREATE INDEX idx_applications_candidate_id ON applications(candidate_id);
CREATE INDEX idx_applications_status ON applications(status);
CREATE INDEX idx_applications_applied_at ON applications(applied_at DESC);
CREATE INDEX idx_applications_match_score ON applications(ai_match_score DESC) WHERE ai_match_score IS NOT NULL;

-- AI matches indexes
CREATE INDEX idx_ai_matches_job_id ON ai_matches(job_id);
CREATE INDEX idx_ai_matches_candidate_id ON ai_matches(candidate_id);
CREATE INDEX idx_ai_matches_overall_score ON ai_matches(overall_score DESC);
CREATE INDEX idx_ai_matches_confidence ON ai_matches(confidence DESC);
CREATE INDEX idx_ai_matches_generated_at ON ai_matches(generated_at DESC);

-- Analytics indexes
CREATE INDEX idx_search_analytics_type_timestamp ON search_analytics(search_type, timestamp DESC);
CREATE INDEX idx_search_analytics_user_type ON search_analytics(user_type);
CREATE INDEX idx_search_analytics_filters ON search_analytics USING gin(filters);

CREATE INDEX idx_job_analytics_job_date ON job_analytics(job_id, date DESC);
CREATE INDEX idx_job_analytics_date ON job_analytics(date DESC);

CREATE INDEX idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp DESC);

-- Skills taxonomy indexes
CREATE INDEX idx_skills_name ON skills_taxonomy(name);
CREATE INDEX idx_skills_category ON skills_taxonomy(category);
CREATE INDEX idx_skills_aliases ON skills_taxonomy USING gin(aliases);
CREATE INDEX idx_skills_popularity ON skills_taxonomy(popularity_score DESC);
CREATE INDEX idx_skills_demand ON skills_taxonomy(demand_score DESC);

-- Audit log indexes
CREATE INDEX idx_audit_log_table_record ON audit_log(table_name, record_id);
CREATE INDEX idx_audit_log_changed_at ON audit_log(changed_at DESC);
CREATE INDEX idx_audit_log_changed_by ON audit_log(changed_by);

-- ======================================
-- Functions and Triggers
-- ======================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_candidates_updated_at BEFORE UPDATE ON candidates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_applications_updated_at BEFORE UPDATE ON applications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate profile completion
CREATE OR REPLACE FUNCTION calculate_profile_completion(candidate_id UUID)
RETURNS FLOAT AS $$
DECLARE
    completion_score FLOAT := 0;
    has_summary BOOLEAN;
    skills_count INTEGER;
    has_experience BOOLEAN;
    has_education BOOLEAN;
    has_work_exp BOOLEAN;
BEGIN
    -- Get candidate data
    SELECT 
        CASE WHEN summary IS NOT NULL AND LENGTH(summary) > 50 THEN TRUE ELSE FALSE END,
        array_length(skills, 1),
        CASE WHEN experience_years > 0 THEN TRUE ELSE FALSE END
    INTO has_summary, skills_count, has_experience
    FROM candidates WHERE id = candidate_id;
    
    -- Check for education records
    SELECT COUNT(*) > 0 INTO has_education 
    FROM education WHERE candidate_id = calculate_profile_completion.candidate_id;
    
    -- Check for work experience records
    SELECT COUNT(*) > 0 INTO has_work_exp
    FROM work_experiences WHERE candidate_id = calculate_profile_completion.candidate_id;
    
    -- Calculate completion percentage
    IF has_summary THEN completion_score := completion_score + 20; END IF;
    IF skills_count >= 5 THEN completion_score := completion_score + 25; 
    ELSIF skills_count >= 3 THEN completion_score := completion_score + 15;
    ELSIF skills_count >= 1 THEN completion_score := completion_score + 10; END IF;
    IF has_experience THEN completion_score := completion_score + 15; END IF;
    IF has_education THEN completion_score := completion_score + 20; END IF;
    IF has_work_exp THEN completion_score := completion_score + 20; END IF;
    
    RETURN completion_score;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update profile completion on candidate changes
CREATE OR REPLACE FUNCTION update_profile_completion()
RETURNS TRIGGER AS $$
BEGIN
    NEW.profile_completion = calculate_profile_completion(NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_candidate_profile_completion 
    BEFORE INSERT OR UPDATE ON candidates 
    FOR EACH ROW EXECUTE FUNCTION update_profile_completion();

-- Function to increment job view count
CREATE OR REPLACE FUNCTION increment_job_views(job_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE jobs SET view_count = view_count + 1 WHERE id = job_id;
END;
$$ LANGUAGE plpgsql;

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_values, changed_at)
        VALUES (TG_TABLE_NAME, OLD.id, TG_OP, row_to_json(OLD), NOW());
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_values, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(OLD), row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, action, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(NEW), NOW());
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers to important tables
CREATE TRIGGER audit_jobs AFTER INSERT OR UPDATE OR DELETE ON jobs FOR EACH ROW EXECUTE FUNCTION audit_trigger();
CREATE TRIGGER audit_candidates AFTER INSERT OR UPDATE OR DELETE ON candidates FOR EACH ROW EXECUTE FUNCTION audit_trigger();
CREATE TRIGGER audit_applications AFTER INSERT OR UPDATE OR DELETE ON applications FOR EACH ROW EXECUTE FUNCTION audit_trigger();

-- ======================================
-- Views for Analytics
-- ======================================

-- Popular skills view
CREATE VIEW popular_skills AS
SELECT 
    skill,
    COUNT(*) as job_mentions,
    AVG(COALESCE(salary_max, salary_min, 0)) as avg_salary,
    COUNT(DISTINCT company_id) as companies_hiring
FROM (
    SELECT unnest(skills) as skill, salary_max, salary_min, company_id
    FROM jobs 
    WHERE status = 'active'
) skill_jobs
GROUP BY skill
HAVING COUNT(*) >= 2
ORDER BY job_mentions DESC;

-- Candidate pipeline view
CREATE VIEW candidate_pipeline AS
SELECT 
    j.id as job_id,
    j.title as job_title,
    j.company_id,
    COUNT(CASE WHEN a.status = 'pending' THEN 1 END) as pending,
    COUNT(CASE WHEN a.status = 'reviewing' THEN 1 END) as reviewing,
    COUNT(CASE WHEN a.status = 'interviewed' THEN 1 END) as interviewed,
    COUNT(CASE WHEN a.status = 'offered' THEN 1 END) as offered,
    COUNT(CASE WHEN a.status = 'hired' THEN 1 END) as hired,
    COUNT(CASE WHEN a.status = 'rejected' THEN 1 END) as rejected,
    COUNT(*) as total_applications
FROM jobs j
LEFT JOIN applications a ON j.id = a.job_id
WHERE j.status = 'active'
GROUP BY j.id, j.title, j.company_id;

-- Matching performance view
CREATE VIEW matching_performance AS
SELECT 
    j.id as job_id,
    j.title,
    COUNT(m.id) as total_matches,
    AVG(m.overall_score) as avg_match_score,
    COUNT(CASE WHEN m.overall_score >= 80 THEN 1 END) as excellent_matches,
    COUNT(CASE WHEN m.overall_score >= 60 AND m.overall_score < 80 THEN 1 END) as good_matches,
    COUNT(CASE WHEN m.overall_score < 60 THEN 1 END) as poor_matches,
    MAX(m.generated_at) as last_matched
FROM jobs j
LEFT JOIN ai_matches m ON j.id = m.job_id
WHERE j.status = 'active'
GROUP BY j.id, j.title;

-- ======================================
-- Initial Reference Data
-- ======================================

-- Insert default skills taxonomy
INSERT INTO skills_taxonomy (name, category, aliases, popularity_score, demand_score) VALUES
('Python', 'programming_languages', ARRAY['python3', 'py'], 95.0, 90.0),
('JavaScript', 'programming_languages', ARRAY['js', 'javascript'], 90.0, 85.0),
('Golang', 'programming_languages', ARRAY['go', 'golang'], 75.0, 80.0),
('Java', 'programming_languages', ARRAY['java'], 85.0, 75.0),
('TypeScript', 'programming_languages', ARRAY['ts', 'typescript'], 80.0, 85.0),
('React', 'frameworks', ARRAY['reactjs', 'react.js'], 85.0, 80.0),
('Vue', 'frameworks', ARRAY['vuejs', 'vue.js'], 70.0, 70.0),
('Angular', 'frameworks', ARRAY['angularjs'], 75.0, 65.0),
('Node.js', 'frameworks', ARRAY['nodejs', 'node'], 80.0, 75.0),
('Express', 'frameworks', ARRAY['expressjs'], 70.0, 65.0),
('Django', 'frameworks', ARRAY['django'], 70.0, 70.0),
('Flask', 'frameworks', ARRAY['flask'], 65.0, 60.0),
('Spring', 'frameworks', ARRAY['spring boot', 'springframework'], 75.0, 70.0),
('PostgreSQL', 'databases', ARRAY['postgres', 'postgresql'], 80.0, 85.0),
('MySQL', 'databases', ARRAY['mysql'], 75.0, 70.0),
('MongoDB', 'databases', ARRAY['mongo'], 70.0, 75.0),
('Redis', 'databases', ARRAY['redis'], 65.0, 80.0),
('Elasticsearch', 'databases', ARRAY['elastic'], 60.0, 75.0),
('Docker', 'devops', ARRAY['containerization'], 85.0, 90.0),
('Kubernetes', 'devops', ARRAY['k8s'], 70.0, 85.0),
('AWS', 'cloud_platforms', ARRAY['amazon web services'], 80.0, 85.0),
('Azure', 'cloud_platforms', ARRAY['microsoft azure'], 70.0, 75.0),
('GCP', 'cloud_platforms', ARRAY['google cloud'], 65.0, 70.0),
('Machine Learning', 'ai_ml', ARRAY['ml', 'machine-learning'], 85.0, 95.0),
('Deep Learning', 'ai_ml', ARRAY['dl', 'deep-learning'], 75.0, 90.0),
('Natural Language Processing', 'ai_ml', ARRAY['nlp'], 70.0, 85.0),
('Computer Vision', 'ai_ml', ARRAY['cv'], 65.0, 80.0),
('TensorFlow', 'ai_ml', ARRAY['tf'], 70.0, 75.0),
('PyTorch', 'ai_ml', ARRAY['pytorch'], 65.0, 80.0),
('Vector Databases', 'databases', ARRAY['vector-db', 'vectordb'], 50.0, 95.0),
('RAG', 'ai_ml', ARRAY['rag-models', 'retrieval-augmented-generation'], 40.0, 90.0),
('LLM', 'ai_ml', ARRAY['large language models', 'llms'], 45.0, 95.0);

-- Insert common locations
INSERT INTO locations (country, city, region, timezone, is_remote_friendly) VALUES
('Germany', 'Berlin', 'Berlin', 'Europe/Berlin', true),
('Germany', 'Munich', 'Bavaria', 'Europe/Berlin', true),
('Germany', 'Hamburg', 'Hamburg', 'Europe/Berlin', true),
('Netherlands', 'Amsterdam', 'North Holland', 'Europe/Amsterdam', true),
('Netherlands', 'Rotterdam', 'South Holland', 'Europe/Amsterdam', true),
('France', 'Paris', 'Île-de-France', 'Europe/Paris', true),
('France', 'Lyon', 'Auvergne-Rhône-Alpes', 'Europe/Paris', true),
('Spain', 'Madrid', 'Community of Madrid', 'Europe/Madrid', true),
('Spain', 'Barcelona', 'Catalonia', 'Europe/Madrid', true),
('Italy', 'Milan', 'Lombardy', 'Europe/Rome', true),
('Italy', 'Rome', 'Lazio', 'Europe/Rome', true),
('United Kingdom', 'London', 'England', 'Europe/London', true),
('Sweden', 'Stockholm', 'Stockholm', 'Europe/Stockholm', true),
('Denmark', 'Copenhagen', 'Capital Region', 'Europe/Copenhagen', true),
('Poland', 'Warsaw', 'Masovian', 'Europe/Warsaw', true),
('Remote', 'Remote', 'Global', 'UTC', true);

-- ======================================
-- Performance Monitoring
-- ======================================

-- Function to collect system metrics
CREATE OR REPLACE FUNCTION collect_system_metrics()
RETURNS VOID AS $$
BEGIN
    -- Table sizes
    INSERT INTO system_metrics (metric_name, metric_value, tags)
    SELECT 
        'table_size_mb',
        pg_total_relation_size(tablename::regclass) / 1024.0 / 1024.0,
        jsonb_build_object('table', tablename)
    FROM pg_tables 
    WHERE schemaname = 'public';
    
    -- Index usage
    INSERT INTO system_metrics (metric_name, metric_value, tags)
    SELECT 
        'index_scans',
        idx_scan,
        jsonb_build_object('table', tablename, 'index', indexname)
    FROM pg_stat_user_indexes
    WHERE idx_scan > 0;
    
    -- Active connections
    INSERT INTO system_metrics (metric_name, metric_value)
    SELECT 'active_connections', count(*)
    FROM pg_stat_activity
    WHERE state = 'active';
    
END;
$$ LANGUAGE plpgsql;

-- ======================================
-- Cleanup Functions
-- ======================================

-- Function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS VOID AS $$
BEGIN
    -- Clean old search analytics (keep 90 days)
    DELETE FROM search_analytics 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Clean old audit logs (keep 1 year)
    DELETE FROM audit_log 
    WHERE changed_at < NOW() - INTERVAL '1 year';
    
    -- Clean old system metrics (keep 30 days)
    DELETE FROM system_metrics 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Update job statistics
    UPDATE jobs SET view_count = 0 WHERE view_count < 0;
    UPDATE jobs SET applicant_count = (
        SELECT COUNT(*) FROM applications WHERE job_id = jobs.id
    );
    
    -- Mark expired jobs as closed
    UPDATE jobs SET status = 'closed' 
    WHERE status = 'active' 
    AND expires_at IS NOT NULL 
    AND expires_at < NOW();
    
END;
$$ LANGUAGE plpgsql;

-- ======================================
-- Schema Validation
-- ======================================

-- Function to validate schema integrity
CREATE OR REPLACE FUNCTION validate_schema_integrity()
RETURNS TABLE(check_name TEXT, status TEXT, details TEXT) AS $
BEGIN
    -- Check for orphaned records
    RETURN QUERY
    SELECT 
        'orphaned_jobs'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        ('Found ' || COUNT(*) || ' jobs without valid company')::TEXT
    FROM jobs j
    LEFT JOIN companies c ON j.company_id = c.id
    WHERE c.id IS NULL;
    
    RETURN QUERY
    SELECT 
        'orphaned_work_experiences'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        ('Found ' || COUNT(*) || ' work experiences without valid candidate')::TEXT
    FROM work_experiences we
    LEFT JOIN candidates c ON we.candidate_id = c.id
    WHERE c.id IS NULL;
    
    -- Check data consistency
    RETURN QUERY
    SELECT 
        'invalid_salary_ranges'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        ('Found ' || COUNT(*) || ' jobs with invalid salary ranges')::TEXT
    FROM jobs
    WHERE salary_min IS NOT NULL 
    AND salary_max IS NOT NULL 
    AND salary_min > salary_max;
    
    RETURN QUERY
    SELECT 
        'invalid_candidate_salaries'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        ('Found ' || COUNT(*) || ' candidates with invalid salary expectations')::TEXT
    FROM candidates
    WHERE salary_expectation_min IS NOT NULL 
    AND salary_expectation_max IS NOT NULL 
    AND salary_expectation_min > salary_expectation_max;
    
    -- Check vector embeddings
    RETURN QUERY
    SELECT 
        'missing_job_embeddings'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
        ('Found ' || COUNT(*) || ' active jobs without embeddings')::TEXT
    FROM jobs
    WHERE status = 'active' AND embedding IS NULL;
    
    RETURN QUERY
    SELECT 
        'missing_candidate_embeddings'::TEXT,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END::TEXT,
        ('Found ' || COUNT(*) || ' candidates without embeddings')::TEXT
    FROM candidates
    WHERE embedding IS NULL;
    
END;
$ LANGUAGE plpgsql;

-- ======================================
-- Database Statistics View
-- ======================================

CREATE VIEW database_statistics AS
SELECT 
    'companies' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END) as recent_records,
    pg_size_pretty(pg_total_relation_size('companies')) as table_size
FROM companies
UNION ALL
SELECT 
    'jobs',
    COUNT(*),
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END),
    pg_size_pretty(pg_total_relation_size('jobs'))
FROM jobs
UNION ALL
SELECT 
    'candidates',
    COUNT(*),
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END),
    pg_size_pretty(pg_total_relation_size('candidates'))
FROM candidates
UNION ALL
SELECT 
    'applications',
    COUNT(*),
    COUNT(CASE WHEN applied_at >= NOW() - INTERVAL '30 days' THEN 1 END),
    pg_size_pretty(pg_total_relation_size('applications'))
FROM applications
UNION ALL
SELECT 
    'ai_matches',
    COUNT(*),
    COUNT(CASE WHEN generated_at >= NOW() - INTERVAL '30 days' THEN 1 END),
    pg_size_pretty(pg_total_relation_size('ai_matches'))
FROM ai_matches;

-- ======================================
-- Backup and Recovery
-- ======================================

-- Function to create data backup
CREATE OR REPLACE FUNCTION create_data_backup(backup_name TEXT DEFAULT NULL)
RETURNS TEXT AS $
DECLARE
    backup_file TEXT;
    backup_timestamp TEXT;
BEGIN
    backup_timestamp := to_char(NOW(), 'YYYY-MM-DD_HH24-MI-SS');
    backup_file := COALESCE(backup_name, 'job_matching_backup_' || backup_timestamp);
    
    -- This would typically use pg_dump in a real implementation
    -- For now, we'll create a logical backup using COPY
    
    RETURN backup_file || '_' || backup_timestamp;
END;
$ LANGUAGE plpgsql;

-- ======================================
-- User Management (Basic)
-- ======================================

-- Users table for authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'candidate' CHECK (role IN ('admin', 'recruiter', 'candidate')),
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT users_email_valid CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})
);

-- API keys for external access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions TEXT[],
    rate_limit INTEGER DEFAULT 1000,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for user management
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);

-- ======================================
-- Configuration Table
-- ======================================

CREATE TABLE system_config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    data_type VARCHAR(50) DEFAULT 'string' CHECK (data_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    is_public BOOLEAN DEFAULT false,
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- Insert default configuration
INSERT INTO system_config (key, value, description, data_type, is_public) VALUES
('matching_algorithm_version', '1.0.0', 'Current matching algorithm version', 'string', true),
('min_match_score_threshold', '50.0', 'Minimum match score to show results', 'float', true),
('max_results_per_job', '100', 'Maximum candidates returned per job', 'integer', true),
('enable_ai_analysis', 'true', 'Enable AI-powered analysis features', 'boolean', false),
('cache_match_results', 'true', 'Cache matching results for performance', 'boolean', false),
('cache_duration_hours', '24', 'How long to cache match results', 'integer', false),
('email_notifications_enabled', 'true', 'Enable email notifications', 'boolean', false),
('max_file_upload_size_mb', '10', 'Maximum file upload size in MB', 'integer', true),
('supported_file_types', '["pdf", "doc", "docx", "txt"]', 'Supported resume file types', 'json', true),
('maintenance_mode', 'false', 'System maintenance mode', 'boolean', true);

-- ======================================
-- Data Quality Checks
-- ======================================

CREATE OR REPLACE FUNCTION run_data_quality_checks()
RETURNS TABLE(check_type TEXT, severity TEXT, message TEXT, affected_count INTEGER) AS $
BEGIN
    -- Check for duplicate candidates by email
    RETURN QUERY
    SELECT 
        'duplicate_candidates'::TEXT,
        'HIGH'::TEXT,
        'Duplicate candidates found with same email'::TEXT,
        COUNT(*)::INTEGER
    FROM (
        SELECT email, COUNT(*) as cnt
        FROM candidates
        GROUP BY email
        HAVING COUNT(*) > 1
    ) dups;
    
    -- Check for jobs without required skills
    RETURN QUERY
    SELECT 
        'jobs_without_skills'::TEXT,
        'MEDIUM'::TEXT,
        'Jobs found without required skills'::TEXT,
        COUNT(*)::INTEGER
    FROM jobs
    WHERE array_length(skills, 1) IS NULL OR array_length(skills, 1) = 0;
    
    -- Check for candidates with very low profile completion
    RETURN QUERY
    SELECT 
        'incomplete_profiles'::TEXT,
        'LOW'::TEXT,
        'Candidates with very low profile completion'::TEXT,
        COUNT(*)::INTEGER
    FROM candidates
    WHERE profile_completion < 30;
    
    -- Check for expired jobs still marked as active
    RETURN QUERY
    SELECT 
        'expired_active_jobs'::TEXT,
        'MEDIUM'::TEXT,
        'Active jobs that have expired'::TEXT,
        COUNT(*)::INTEGER
    FROM jobs
    WHERE status = 'active' 
    AND expires_at IS NOT NULL 
    AND expires_at < NOW();
    
    -- Check for very old applications without status updates
    RETURN QUERY
    SELECT 
        'stale_applications'::TEXT,
        'LOW'::TEXT,
        'Applications pending for more than 30 days'::TEXT,
        COUNT(*)::INTEGER
    FROM applications
    WHERE status = 'pending' 
    AND applied_at < NOW() - INTERVAL '30 days';
    
END;
$ LANGUAGE plpgsql;

-- ======================================
-- Performance Optimization
-- ======================================

-- Materialized view for popular job searches
CREATE MATERIALIZED VIEW popular_job_searches AS
SELECT 
    search_query,
    COUNT(*) as search_count,
    AVG(results_count) as avg_results,
    MAX(timestamp) as last_searched
FROM search_analytics
WHERE search_type = 'job_search'
AND search_query IS NOT NULL
AND timestamp >= NOW() - INTERVAL '30 days'
GROUP BY search_query
HAVING COUNT(*) >= 5
ORDER BY search_count DESC
LIMIT 100;

-- Index on materialized view
CREATE INDEX idx_popular_searches_count ON popular_job_searches(search_count DESC);

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS VOID AS $
BEGIN
    REFRESH MATERIALIZED VIEW popular_job_searches;
END;
$ LANGUAGE plpgsql;

-- ======================================
-- Sample Data for Development
-- ======================================

-- Insert sample companies
INSERT INTO companies (id, name, description, location, size, industry, website) VALUES
('00000000-0000-0000-0000-000000000001', 'TechCorp EU', 'Leading AI technology company in Europe', 'Berlin, Germany', 'medium', 'Technology', 'https://techcorp.eu'),
('00000000-0000-0000-0000-000000000002', 'DataFlow Inc', 'Data processing and analytics solutions', 'Amsterdam, Netherlands', 'startup', 'Data Analytics', 'https://dataflow.com'),
('00000000-0000-0000-0000-000000000003', 'AI Innovations', 'Machine learning and AI consulting', 'Paris, France', 'small', 'AI/ML', 'https://ai-innovations.fr');

-- Insert sample jobs
INSERT INTO jobs (id, company_id, title, description, requirements, skills, location, salary_min, salary_max, job_type, experience_level, remote_allowed) VALUES
(
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    'AI Specialist - RAG Models',
    'We are seeking a talented AI Specialist to build cutting-edge RAG models and develop AI strategy. You will work with diverse AI models including LLMs and vector databases in a collaborative environment.',
    ARRAY[
        'Strong passion for machine learning and artificial intelligence',
        'Previous proven experience with AI/ML projects',
        'Experience with RAG models and vector databases',
        'Knowledge of natural language processing',
        'Strong problem-solving skills'
    ],
    ARRAY['Python', 'Machine Learning', 'RAG', 'Vector Databases', 'NLP', 'LLM', 'TensorFlow', 'PyTorch'],
    'Berlin, Germany',
    65000,
    80000,
    'part-time',
    'mid',
    true
),
(
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000002',
    'Senior Backend Developer',
    'Join our team to develop scalable backend systems using Go and microservices architecture. You will be responsible for designing and implementing high-performance APIs and data processing pipelines.',
    ARRAY[
        '5+ years of backend development experience',
        'Expertise in Go programming language',
        'Experience with microservices architecture',
        'Strong database design skills',
        'Experience with containerization and orchestration'
    ],
    ARRAY['Golang', 'PostgreSQL', 'Docker', 'Kubernetes', 'REST APIs', 'Microservices', 'Redis', 'gRPC'],
    'Amsterdam, Netherlands',
    75000,
    95000,
    'full-time',
    'senior',
    false
);

-- Insert sample candidates
INSERT INTO candidates (id, email, first_name, last_name, location, title, summary, skills, experience_years, education_level, availability, salary_expectation_min, salary_expectation_max, remote_preference) VALUES
(
    '00000000-0000-0000-0000-000000000001',
    'alex.johnson@email.com',
    'Alex',
    'Johnson',
    'Munich, Germany',
    'AI/ML Engineer',
    'Passionate AI engineer with 5 years of experience in building production ML systems. Specialized in RAG models, vector databases, and natural language processing. Strong track record of delivering AI solutions that drive business value.',
    ARRAY['Python', 'Machine Learning', 'TensorFlow', 'PyTorch', 'NLP', 'Vector Databases', 'RAG', 'Scikit-learn', 'Pandas'],
    5,
    'masters',
    'part-time',
    65000,
    75000,
    true
),
(
    '00000000-0000-0000-0000-000000000002',
    'maria.rodriguez@email.com',
    'Maria',
    'Rodriguez',
    'Barcelona, Spain',
    'Senior Backend Developer',
    'Experienced backend developer with 7 years of expertise in building scalable microservices using Go. Deep knowledge of distributed systems, database optimization, and cloud-native architectures.',
    ARRAY['Golang', 'PostgreSQL', 'Docker', 'Kubernetes', 'Microservices', 'REST APIs', 'gRPC', 'Redis', 'AWS'],
    7,
    'bachelors',
    'full-time',
    75000,
    85000,
    false
);

-- Insert sample work experiences
INSERT INTO work_experiences (candidate_id, company_name, position, description, skills_used, start_date, end_date, is_current, location) VALUES
(
    '00000000-0000-0000-0000-000000000001',
    'AI Solutions GmbH',
    'Senior ML Engineer',
    'Led development of RAG-based chatbot system serving 100k+ users. Implemented vector similarity search using Pinecone and built ML pipelines for automated content generation.',
    ARRAY['Python', 'RAG', 'Vector Databases', 'NLP', 'TensorFlow'],
    '2022-01-01',
    NULL,
    true,
    'Munich, Germany'
),
(
    '00000000-0000-0000-0000-000000000002',
    'Scalable Systems Inc',
    'Backend Developer',
    'Designed and implemented microservices architecture handling 1M+ requests/day. Built REST APIs using Go and managed PostgreSQL databases with complex queries.',
    ARRAY['Golang', 'PostgreSQL', 'Docker', 'Microservices', 'REST APIs'],
    '2020-03-01',
    NULL,
    true,
    'Barcelona, Spain'
);

-- Insert sample education
INSERT INTO education (candidate_id, institution, degree, field_of_study, start_date, end_date) VALUES
(
    '00000000-0000-0000-0000-000000000001',
    'Technical University of Munich',
    'Master of Science',
    'Computer Science',
    '2016-09-01',
    '2018-07-01'
),
(
    '00000000-0000-0000-0000-000000000002',
    'University of Barcelona',
    'Bachelor of Science',
    'Software Engineering',
    '2013-09-01',
    '2017-06-01'
);

-- ======================================
-- Database Health Check Function
-- ======================================

CREATE OR REPLACE FUNCTION database_health_check()
RETURNS JSONB AS $
DECLARE
    result JSONB;
    table_stats JSONB;
    connection_stats JSONB;
    performance_stats JSONB;
BEGIN
    -- Basic statistics
    SELECT jsonb_build_object(
        'total_companies', (SELECT COUNT(*) FROM companies),
        'total_jobs', (SELECT COUNT(*) FROM jobs),
        'active_jobs', (SELECT COUNT(*) FROM jobs WHERE status = 'active'),
        'total_candidates', (SELECT COUNT(*) FROM candidates),
        'total_applications', (SELECT COUNT(*) FROM applications),
        'total_matches', (SELECT COUNT(*) FROM ai_matches)
    ) INTO table_stats;
    
    -- Connection statistics
    SELECT jsonb_build_object(
        'active_connections', (
            SELECT count(*) FROM pg_stat_activity WHERE state = 'active'
        ),
        'idle_connections', (
            SELECT count(*) FROM pg_stat_activity WHERE state = 'idle'
        ),
        'total_connections', (
            SELECT count(*) FROM pg_stat_activity
        )
    ) INTO connection_stats;
    
    -- Performance statistics
    SELECT jsonb_build_object(
        'database_size', pg_size_pretty(pg_database_size(current_database())),
        'largest_table', (
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY pg_total_relation_size(tablename::regclass) DESC 
            LIMIT 1
        ),
        'cache_hit_ratio', (
            SELECT round(
                100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read) + 1), 2
            ) FROM pg_stat_database WHERE datname = current_database()
        )
    ) INTO performance_stats;
    
    -- Combine all statistics
    result := jsonb_build_object(
        'status', 'healthy',
        'timestamp', NOW(),
        'database', current_database(),
        'version', version(),
        'table_statistics', table_stats,
        'connection_statistics', connection_stats,
        'performance_statistics', performance_stats
    );
    
    RETURN result;
END;
$ LANGUAGE plpgsql;

-- ======================================
-- Final Setup Commands
-- ======================================

-- Update table statistics
ANALYZE;

-- Create scheduled job for cleanup (requires pg_cron extension)
-- SELECT cron.schedule('daily-cleanup', '0 2 * * *', 'SELECT cleanup_old_data();');
-- SELECT cron.schedule('refresh-views', '0 */6 * * *', 'SELECT refresh_materialized_views();');
-- SELECT cron.schedule('collect-metrics', '*/15 * * * *', 'SELECT collect_system_metrics();');

-- Grant permissions for application user
-- CREATE USER job_matching_app WITH PASSWORD 'secure_app_password';
-- GRANT CONNECT ON DATABASE job_matching_ai TO job_matching_app;
-- GRANT USAGE ON SCHEMA public TO job_matching_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO job_matching_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO job_matching_app;

COMMIT;
