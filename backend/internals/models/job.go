package models

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// JobPosting represents a job posting in the system
type JobPosting struct {
	ID              uuid.UUID         `json:"id" db:"id"`
	CompanyID       uuid.UUID         `json:"company_id" db:"company_id"`
	Title           string            `json:"title" db:"title" validate:"required,min=3,max=255"`
	Description     string            `json:"description" db:"description" validate:"required,min=10"`
	Requirements    pq.StringArray    `json:"requirements" db:"requirements"`
	Skills          pq.StringArray    `json:"skills" db:"skills" validate:"required,min=1"`
	Location        string            `json:"location" db:"location" validate:"required"`
	SalaryMin       *int              `json:"salary_min" db:"salary_min"`
	SalaryMax       *int              `json:"salary_max" db:"salary_max"`
	Currency        string            `json:"currency" db:"currency"`
	JobType         string            `json:"job_type" db:"job_type" validate:"required,oneof=full-time part-time contract internship"`
	WorkHours       *string           `json:"work_hours" db:"work_hours"`
	ExperienceLevel string            `json:"experience_level" db:"experience_level"`
	RemoteAllowed   bool              `json:"remote_allowed" db:"remote_allowed"`
	Status          string            `json:"status" db:"status"`
	PostedAt        time.Time         `json:"posted_at" db:"posted_at"`
	ExpiresAt       *time.Time        `json:"expires_at" db:"expires_at"`
	ApplicantCount  int               `json:"applicant_count" db:"applicant_count"`
	ViewCount       int               `json:"view_count" db:"view_count"`
	Embedding       []float32         `json:"-" db:"embedding"` // Hidden from JSON
	AIScore         float64           `json:"ai_score" db:"ai_score"`
	CreatedAt       time.Time         `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at" db:"updated_at"`
	
	// Joined fields (not in database)
	Company         *Company          `json:"company,omitempty"`
	MatchScore      float64           `json:"match_score,omitempty"`
	MatchDetails    *MatchBreakdown   `json:"match_details,omitempty"`
}

// Company represents basic company information
type Company struct {
	ID          uuid.UUID `json:"id" db:"id"`
	Name        string    `json:"name" db:"name"`
	Description *string   `json:"description" db:"description"`
	Website     *string   `json:"website" db:"website"`
	Location    string    `json:"location" db:"location"`
	Size        string    `json:"size" db:"size"`
	Industry    string    `json:"industry" db:"industry"`
	LogoURL     *string   `json:"logo_url" db:"logo_url"`
}

// JobCreateRequest represents the request to create a new job
type JobCreateRequest struct {
	CompanyID       uuid.UUID      `json:"company_id" validate:"required"`
	Title           string         `json:"title" validate:"required,min=3,max=255"`
	Description     string         `json:"description" validate:"required,min=10"`
	Requirements    []string       `json:"requirements"`
	Skills          []string       `json:"skills" validate:"required,min=1"`
	Location        string         `json:"location" validate:"required"`
	SalaryMin       *int           `json:"salary_min" validate:"omitempty,min=0"`
	SalaryMax       *int           `json:"salary_max" validate:"omitempty,min=0"`
	Currency        string         `json:"currency"`
	JobType         string         `json:"job_type" validate:"required,oneof=full-time part-time contract internship"`
	WorkHours       *string        `json:"work_hours"`
	ExperienceLevel string         `json:"experience_level"`
	RemoteAllowed   bool           `json:"remote_allowed"`
	ExpiresAt       *time.Time     `json:"expires_at"`
}

// JobUpdateRequest represents the request to update a job
type JobUpdateRequest struct {
	Title           *string        `json:"title" validate:"omitempty,min=3,max=255"`
	Description     *string        `json:"description" validate:"omitempty,min=10"`
	Requirements    *[]string      `json:"requirements"`
	Skills          *[]string      `json:"skills" validate:"omitempty,min=1"`
	Location        *string        `json:"location"`
	SalaryMin       *int           `json:"salary_min" validate:"omitempty,min=0"`
	SalaryMax       *int           `json:"salary_max" validate:"omitempty,min=0"`
	Currency        *string        `json:"currency"`
	JobType         *string        `json:"job_type" validate:"omitempty,oneof=full-time part-time contract internship"`
	WorkHours       *string        `json:"work_hours"`
	ExperienceLevel *string        `json:"experience_level"`
	RemoteAllowed   *bool          `json:"remote_allowed"`
	Status          *string        `json:"status" validate:"omitempty,oneof=active paused closed"`
	ExpiresAt       *time.Time     `json:"expires_at"`
}

// JobSearchRequest represents job search parameters
type JobSearchRequest struct {
	Query           string    `json:"query" form:"query"`
	Skills          []string  `json:"skills" form:"skills"`
	Location        string    `json:"location" form:"location"`
	JobType         string    `json:"job_type" form:"job_type"`
	RemoteAllowed   *bool     `json:"remote_allowed" form:"remote_allowed"`
	SalaryMin       *int      `json:"salary_min" form:"salary_min"`
	SalaryMax       *int      `json:"salary_max" form:"salary_max"`
	ExperienceLevel string    `json:"experience_level" form:"experience_level"`
	CompanyID       *uuid.UUID `json:"company_id" form:"company_id"`
	Limit           int       `json:"limit" form:"limit" validate:"omitempty,min=1,max=100"`
	Offset          int       `json:"offset" form:"offset" validate:"omitempty,min=0"`
	SortBy          string    `json:"sort_by" form:"sort_by" validate:"omitempty,oneof=created_at salary_max match_score applicant_count"`
	SortOrder       string    `json:"sort_order" form:"sort_order" validate:"omitempty,oneof=asc desc"`
}

// JobAnalytics represents job analytics data
type JobAnalytics struct {
	JobID           uuid.UUID `json:"job_id" db:"job_id"`
	Views           int       `json:"views" db:"views"`
	Applications    int       `json:"applications" db:"applications"`
	MatchesFound    int       `json:"matches_found" db:"matches_found"`
	AvgMatchScore   float64   `json:"avg_match_score" db:"avg_match_score"`
	TopSkills       []string  `json:"top_skills" db:"top_skills"`
	ApplicationRate float64   `json:"application_rate"`
	PopularityScore float64   `json:"popularity_score"`
}

// JobStatus constants
const (
	JobStatusActive = "active"
	JobStatusPaused = "paused"
	JobStatusClosed = "closed"
)

// JobType constants
const (
	JobTypeFullTime   = "full-time"
	JobTypePartTime   = "part-time"
	JobTypeContract   = "contract"
	JobTypeInternship = "internship"
)

// ExperienceLevel constants
const (
	ExperienceLevelEntry     = "entry"
	ExperienceLevelJunior    = "junior"
	ExperienceLevelMid       = "mid"
	ExperienceLevelSenior    = "senior"
	ExperienceLevelLead      = "lead"
	ExperienceLevelPrincipal = "principal"
)

// Validate validates the job posting data
func (j *JobPosting) Validate() error {
	if j.Title == "" {
		return fmt.Errorf("title is required")
	}
	
	if j.Description == "" {
		return fmt.Errorf("description is required")
	}
	
	if len(j.Skills) == 0 {
		return fmt.Errorf("at least one skill is required")
	}
	
	if j.Location == "" {
		return fmt.Errorf("location is required")
	}
	
	if j.SalaryMin != nil && j.SalaryMax != nil && *j.SalaryMin > *j.SalaryMax {
		return fmt.Errorf("salary_min cannot be greater than salary_max")
	}
	
	return nil
}

// IsActive returns true if the job is active and not expired
func (j *JobPosting) IsActive() bool {
	if j.Status != JobStatusActive {
		return false
	}
	
	if j.ExpiresAt != nil && j.ExpiresAt.Before(time.Now()) {
		return false
	}
	
	return true
}

// GetSalaryRange returns formatted salary range
func (j *JobPosting) GetSalaryRange() string {
	if j.SalaryMin == nil && j.SalaryMax == nil {
		return "Salary not specified"
	}
	
	currency := j.Currency
	if currency == "" {
		currency = "EUR"
	}
	
	if j.SalaryMin != nil && j.SalaryMax != nil {
		return fmt.Sprintf("%s %d - %d", currency, *j.SalaryMin, *j.SalaryMax)
	} else if j.SalaryMin != nil {
		return fmt.Sprintf("%s %d+", currency, *j.SalaryMin)
	} else {
		return fmt.Sprintf("Up to %s %d", currency, *j.SalaryMax)
	}
}

// ToCreateRequest converts JobPosting to JobCreateRequest
func (j *JobPosting) ToCreateRequest() JobCreateRequest {
	return JobCreateRequest{
		CompanyID:       j.CompanyID,
		Title:           j.Title,
		Description:     j.Description,
		Requirements:    []string(j.Requirements),
		Skills:          []string(j.Skills),
		Location:        j.Location,
		SalaryMin:       j.SalaryMin,
		SalaryMax:       j.SalaryMax,
		Currency:        j.Currency,
		JobType:         j.JobType,
		WorkHours:       j.WorkHours,
		ExperienceLevel: j.ExperienceLevel,
		RemoteAllowed:   j.RemoteAllowed,
		ExpiresAt:       j.ExpiresAt,
	}
}

// ApplyDefaults sets default values for a new job posting
func (j *JobPosting) ApplyDefaults() {
	if j.ID == uuid.Nil {
		j.ID = uuid.New()
	}
	
	if j.Currency == "" {
		j.Currency = "EUR"
	}
	
	if j.Status == "" {
		j.Status = JobStatusActive
	}
	
	now := time.Now()
	if j.PostedAt.IsZero() {
		j.PostedAt = now
	}
	
	if j.CreatedAt.IsZero() {
		j.CreatedAt = now
	}
	
	j.UpdatedAt = now
}

// Float32Array is a custom type for handling float32 arrays in PostgreSQL
type Float32Array []float32

// Scan implements the sql.Scanner interface
func (a *Float32Array) Scan(value interface{}) error {
	if value == nil {
		*a = nil
		return nil
	}
	
	switch v := value.(type) {
	case []byte:
		return json.Unmarshal(v, a)
	case string:
		return json.Unmarshal([]byte(v), a)
	default:
		return fmt.Errorf("cannot scan %T into Float32Array", value)
	}
}

// Value implements the driver.Valuer interface
func (a Float32Array) Value() (driver.Value, error) {
	if a == nil {
		return nil, nil
	}
	return json.Marshal(a)
}
