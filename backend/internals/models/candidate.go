package models

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// Candidate represents a job candidate in the system
type Candidate struct {
	ID                   uuid.UUID      `json:"id" db:"id"`
	Email                string         `json:"email" db:"email" validate:"required,email"`
	FirstName            string         `json:"first_name" db:"first_name" validate:"required,min=2,max=100"`
	LastName             string         `json:"last_name" db:"last_name" validate:"required,min=2,max=100"`
	Phone                *string        `json:"phone" db:"phone"`
	Location             string         `json:"location" db:"location"`
	Title                string         `json:"title" db:"title"`
	Summary              *string        `json:"summary" db:"summary"`
	Skills               pq.StringArray `json:"skills" db:"skills" validate:"required,min=1"`
	ExperienceYears      int            `json:"experience_years" db:"experience_years"`
	EducationLevel       string         `json:"education_level" db:"education_level"`
	EducationField       *string        `json:"education_field" db:"education_field"`
	Availability         string         `json:"availability" db:"availability"`
	SalaryExpectationMin *int           `json:"salary_expectation_min" db:"salary_expectation_min"`
	SalaryExpectationMax *int           `json:"salary_expectation_max" db:"salary_expectation_max"`
	Currency             string         `json:"currency" db:"currency"`
	RemotePreference     bool           `json:"remote_preference" db:"remote_preference"`
	WillingToRelocate    bool           `json:"willing_to_relocate" db:"willing_to_relocate"`
	Embedding            []float32      `json:"-" db:"embedding"`
	AIScore              float64        `json:"ai_score" db:"ai_score"`
	ProfileCompletion    float64        `json:"profile_completion" db:"profile_completion"`
	LastActive           time.Time      `json:"last_active" db:"last_active"`
	CreatedAt            time.Time      `json:"created_at" db:"created_at"`
	UpdatedAt            time.Time      `json:"updated_at" db:"updated_at"`
	
	// Joined fields (not in database)
	WorkExperiences      []WorkExperience `json:"work_experiences,omitempty"`
	Education            []Education      `json:"education,omitempty"`
	MatchScore           float64          `json:"match_score,omitempty"`
	MatchDetails         *MatchBreakdown  `json:"match_details,omitempty"`
}

// WorkExperience represents a candidate's work experience
type WorkExperience struct {
	ID          uuid.UUID      `json:"id" db:"id"`
	CandidateID uuid.UUID      `json:"candidate_id" db:"candidate_id"`
	CompanyName string         `json:"company_name" db:"company_name" validate:"required"`
	Position    string         `json:"position" db:"position" validate:"required"`
	Description *string        `json:"description" db:"description"`
	SkillsUsed  pq.StringArray `json:"skills_used" db:"skills_used"`
	StartDate   time.Time      `json:"start_date" db:"start_date" validate:"required"`
	EndDate     *time.Time     `json:"end_date" db:"end_date"`
	IsCurrent   bool           `json:"is_current" db:"is_current"`
	Location    *string        `json:"location" db:"location"`
	CreatedAt   time.Time      `json:"created_at" db:"created_at"`
}

// Education represents a candidate's education
type Education struct {
	ID            uuid.UUID  `json:"id" db:"id"`
	CandidateID   uuid.UUID  `json:"candidate_id" db:"candidate_id"`
	Institution   string     `json:"institution" db:"institution" validate:"required"`
	Degree        string     `json:"degree" db:"degree" validate:"required"`
	FieldOfStudy  *string    `json:"field_of_study" db:"field_of_study"`
	StartDate     *time.Time `json:"start_date" db:"start_date"`
	EndDate       *time.Time `json:"end_date" db:"end_date"`
	GPA           *float64   `json:"gpa" db:"gpa" validate:"omitempty,min=0,max=4"`
	Description   *string    `json:"description" db:"description"`
	CreatedAt     time.Time  `json:"created_at" db:"created_at"`
}

// CandidateCreateRequest represents the request to create a new candidate
type CandidateCreateRequest struct {
	Email                string    `json:"email" validate:"required,email"`
	FirstName            string    `json:"first_name" validate:"required,min=2,max=100"`
	LastName             string    `json:"last_name" validate:"required,min=2,max=100"`
	Phone                *string   `json:"phone"`
	Location             string    `json:"location"`
	Title                string    `json:"title"`
	Summary              *string   `json:"summary"`
	Skills               []string  `json:"skills" validate:"required,min=1"`
	ExperienceYears      int       `json:"experience_years" validate:"min=0"`
	EducationLevel       string    `json:"education_level"`
	EducationField       *string   `json:"education_field"`
	Availability         string    `json:"availability"`
	SalaryExpectationMin *int      `json:"salary_expectation_min" validate:"omitempty,min=0"`
	SalaryExpectationMax *int      `json:"salary_expectation_max" validate:"omitempty,min=0"`
	Currency             string    `json:"currency"`
	RemotePreference     bool      `json:"remote_preference"`
	WillingToRelocate    bool      `json:"willing_to_relocate"`
}

// CandidateUpdateRequest represents the request to update a candidate
type CandidateUpdateRequest struct {
	FirstName            *string   `json:"first_name" validate:"omitempty,min=2,max=100"`
	LastName             *string   `json:"last_name" validate:"omitempty,min=2,max=100"`
	Phone                *string   `json:"phone"`
	Location             *string   `json:"location"`
	Title                *string   `json:"title"`
	Summary              *string   `json:"summary"`
	Skills               *[]string `json:"skills" validate:"omitempty,min=1"`
	ExperienceYears      *int      `json:"experience_years" validate:"omitempty,min=0"`
	EducationLevel       *string   `json:"education_level"`
	EducationField       *string   `json:"education_field"`
	Availability         *string   `json:"availability"`
	SalaryExpectationMin *int      `json:"salary_expectation_min" validate:"omitempty,min=0"`
	SalaryExpectationMax *int      `json:"salary_expectation_max" validate:"omitempty,min=0"`
	Currency             *string   `json:"currency"`
	RemotePreference     *bool     `json:"remote_preference"`
	WillingToRelocate    *bool     `json:"willing_to_relocate"`
}

// CandidateSearchRequest represents candidate search parameters
type CandidateSearchRequest struct {
	Query           string     `json:"query" form:"query"`
	Skills          []string   `json:"skills" form:"skills"`
	Location        string     `json:"location" form:"location"`
	Availability    string     `json:"availability" form:"availability"`
	ExperienceMin   *int       `json:"experience_min" form:"experience_min"`
	ExperienceMax   *int       `json:"experience_max" form:"experience_max"`
	SalaryMin       *int       `json:"salary_min" form:"salary_min"`
	SalaryMax       *int       `json:"salary_max" form:"salary_max"`
	EducationLevel  string     `json:"education_level" form:"education_level"`
	RemoteOnly      *bool      `json:"remote_only" form:"remote_only"`
	WillingToRelocate *bool    `json:"willing_to_relocate" form:"willing_to_relocate"`
	Limit           int        `json:"limit" form:"limit" validate:"omitempty,min=1,max=100"`
	Offset          int        `json:"offset" form:"offset" validate:"omitempty,min=0"`
	SortBy          string     `json:"sort_by" form:"sort_by" validate:"omitempty,oneof=created_at experience_years match_score profile_completion"`
	SortOrder       string     `json:"sort_order" form:"sort_order" validate:"omitempty,oneof=asc desc"`
}

// WorkExperienceCreateRequest represents request to add work experience
type WorkExperienceCreateRequest struct {
	CandidateID uuid.UUID `json:"candidate_id" validate:"required"`
	CompanyName string    `json:"company_name" validate:"required"`
	Position    string    `json:"position" validate:"required"`
	Description *string   `json:"description"`
	SkillsUsed  []string  `json:"skills_used"`
	StartDate   time.Time `json:"start_date" validate:"required"`
	EndDate     *time.Time `json:"end_date"`
	IsCurrent   bool      `json:"is_current"`
	Location    *string   `json:"location"`
}

// EducationCreateRequest represents request to add education
type EducationCreateRequest struct {
	CandidateID  uuid.UUID  `json:"candidate_id" validate:"required"`
	Institution  string     `json:"institution" validate:"required"`
	Degree       string     `json:"degree" validate:"required"`
	FieldOfStudy *string    `json:"field_of_study"`
	StartDate    *time.Time `json:"start_date"`
	EndDate      *time.Time `json:"end_date"`
	GPA          *float64   `json:"gpa" validate:"omitempty,min=0,max=4"`
	Description  *string    `json:"description"`
}

// Availability constants
const (
	AvailabilityFullTime = "full-time"
	AvailabilityPartTime = "part-time"
	AvailabilityContract = "contract"
	AvailabilityFreelance = "freelance"
)

// EducationLevel constants
const (
	EducationHighSchool = "high_school"
	EducationBachelors  = "bachelors"
	EducationMasters    = "masters"
	EducationDoctorate  = "doctorate"
	EducationCertificate = "certificate"
	EducationDiploma    = "diploma"
)

// Validate validates the candidate data
func (c *Candidate) Validate() error {
	if c.Email == "" {
		return fmt.Errorf("email is required")
	}
	
	if c.FirstName == "" {
		return fmt.Errorf("first name is required")
	}
	
	if c.LastName == "" {
		return fmt.Errorf("last name is required")
	}
	
	if len(c.Skills) == 0 {
		return fmt.Errorf("at least one skill is required")
	}
	
	if c.SalaryExpectationMin != nil && c.SalaryExpectationMax != nil && 
		*c.SalaryExpectationMin > *c.SalaryExpectationMax {
		return fmt.Errorf("salary_expectation_min cannot be greater than salary_expectation_max")
	}
	
	return nil
}

// GetFullName returns the candidate's full name
func (c *Candidate) GetFullName() string {
	return c.FirstName + " " + c.LastName
}

// GetSalaryExpectation returns formatted salary expectation
func (c *Candidate) GetSalaryExpectation() string {
	if c.SalaryExpectationMin == nil && c.SalaryExpectationMax == nil {
		return "Salary expectations not specified"
	}
	
	currency := c.Currency
	if currency == "" {
		currency = "EUR"
	}
	
	if c.SalaryExpectationMin != nil && c.SalaryExpectationMax != nil {
		return fmt.Sprintf("%s %d - %d", currency, *c.SalaryExpectationMin, *c.SalaryExpectationMax)
	} else if c.SalaryExpectationMin != nil {
		return fmt.Sprintf("%s %d+", currency, *c.SalaryExpectationMin)
	} else {
		return fmt.Sprintf("Up to %s %d", currency, *c.SalaryExpectationMax)
	}
}

// GetExperienceLevel returns experience level based on years
func (c *Candidate) GetExperienceLevel() string {
	years := c.ExperienceYears
	
	if years <= 1 {
		return ExperienceLevelEntry
	} else if years <= 3 {
		return ExperienceLevelJunior
	} else if years <= 7 {
		return ExperienceLevelMid
	} else if years <= 12 {
		return ExperienceLevelSenior
	} else {
		return ExperienceLevelPrincipal
	}
}

// CalculateProfileCompletion calculates profile completion percentage
func (c *Candidate) CalculateProfileCompletion() float64 {
	score := 0.0
	total := 10.0 // Total possible points
	
	// Basic info (required fields get points automatically)
	score += 2.0 // Email, FirstName, LastName, Skills are required
	
	// Optional but important fields
	if c.Phone != nil && *c.Phone != "" {
		score += 0.5
	}
	
	if c.Location != "" {
		score += 0.5
	}
	
	if c.Title != "" {
		score += 1.0
	}
	
	if c.Summary != nil && *c.Summary != "" && len(*c.Summary) >= 50 {
		score += 1.5
	}
	
	if c.ExperienceYears > 0 {
		score += 1.0
	}
	
	if c.EducationLevel != "" {
		score += 0.5
	}
	
	if len(c.Skills) >= 5 {
		score += 1.0
	} else if len(c.Skills) >= 3 {
		score += 0.5
	}
	
	if c.SalaryExpectationMin != nil || c.SalaryExpectationMax != nil {
		score += 0.5
	}
	
	// Experience and education records
	if len(c.WorkExperiences) > 0 {
		score += 1.0
	}
	
	if len(c.Education) > 0 {
		score += 0.5
	}
	
	return (score / total) * 100
}

// IsActiveCandidate returns true if candidate was active recently
func (c *Candidate) IsActiveCandidate() bool {
	thirtyDaysAgo := time.Now().AddDate(0, 0, -30)
	return c.LastActive.After(thirtyDaysAgo)
}

// ApplyDefaults sets default values for a new candidate
func (c *Candidate) ApplyDefaults() {
	if c.ID == uuid.Nil {
		c.ID = uuid.New()
	}
	
	if c.Currency == "" {
		c.Currency = "EUR"
	}
	
	if c.Availability == "" {
		c.Availability = AvailabilityFullTime
	}
	
	now := time.Now()
	if c.CreatedAt.IsZero() {
		c.CreatedAt = now
	}
	
	if c.LastActive.IsZero() {
		c.LastActive = now
	}
	
	c.UpdatedAt = now
	c.ProfileCompletion = c.CalculateProfileCompletion()
}

// ToCreateRequest converts Candidate to CandidateCreateRequest
func (c *Candidate) ToCreateRequest() CandidateCreateRequest {
	return CandidateCreateRequest{
		Email:                c.Email,
		FirstName:            c.FirstName,
		LastName:             c.LastName,
		Phone:                c.Phone,
		Location:             c.Location,
		Title:                c.Title,
		Summary:              c.Summary,
		Skills:               []string(c.Skills),
		ExperienceYears:      c.ExperienceYears,
		EducationLevel:       c.EducationLevel,
		EducationField:       c.EducationField,
		Availability:         c.Availability,
		SalaryExpectationMin: c.SalaryExpectationMin,
		SalaryExpectationMax: c.SalaryExpectationMax,
		Currency:             c.Currency,
		RemotePreference:     c.RemotePreference,
		WillingToRelocate:    c.WillingToRelocate,
	}
}

// Validate validates work experience data
func (we *WorkExperience) Validate() error {
	if we.CompanyName == "" {
		return fmt.Errorf("company name is required")
	}
	
	if we.Position == "" {
		return fmt.Errorf("position is required")
	}
	
	if we.StartDate.IsZero() {
		return fmt.Errorf("start date is required")
	}
	
	if we.EndDate != nil && we.EndDate.Before(we.StartDate) {
		return fmt.Errorf("end date cannot be before start date")
	}
	
	if we.IsCurrent && we.EndDate != nil {
		return fmt.Errorf("current position cannot have an end date")
	}
	
	return nil
}

// GetDuration returns the duration of work experience
func (we *WorkExperience) GetDuration() time.Duration {
	endDate := time.Now()
	if we.EndDate != nil {
		endDate = *we.EndDate
	}
	
	return endDate.Sub(we.StartDate)
}

// GetDurationInMonths returns work experience duration in months
func (we *WorkExperience) GetDurationInMonths() int {
	duration := we.GetDuration()
	return int(duration.Hours() / 24 / 30) // Approximate months
}

// ApplyDefaults sets default values for work experience
func (we *WorkExperience) ApplyDefaults() {
	if we.ID == uuid.Nil {
		we.ID = uuid.New()
	}
	
	if we.CreatedAt.IsZero() {
		we.CreatedAt = time.Now()
	}
}

// Validate validates education data
func (e *Education) Validate() error {
	if e.Institution == "" {
		return fmt.Errorf("institution is required")
	}
	
	if e.Degree == "" {
		return fmt.Errorf("degree is required")
	}
	
	if e.StartDate != nil && e.EndDate != nil && e.EndDate.Before(*e.StartDate) {
		return fmt.Errorf("end date cannot be before start date")
	}
	
	if e.GPA != nil && (*e.GPA < 0 || *e.GPA > 4) {
		return fmt.Errorf("GPA must be between 0 and 4")
	}
	
	return nil
}

// ApplyDefaults sets default values for education
func (e *Education) ApplyDefaults() {
	if e.ID == uuid.Nil {
		e.ID = uuid.New()
	}
	
	if e.CreatedAt.IsZero() {
		e.CreatedAt = time.Now()
	}
}

// CandidateProfile represents a complete candidate profile with all related data
type CandidateProfile struct {
	Candidate       Candidate        `json:"candidate"`
	WorkExperiences []WorkExperience `json:"work_experiences"`
	Education       []Education      `json:"education"`
	TotalExperience int              `json:"total_experience_months"`
	ProfileStats    ProfileStats     `json:"profile_stats"`
}

// ProfileStats represents candidate profile statistics
type ProfileStats struct {
	ProfileCompletion   float64   `json:"profile_completion"`
	SkillCount          int       `json:"skill_count"`
	ExperienceCount     int       `json:"experience_count"`
	EducationCount      int       `json:"education_count"`
	LastUpdated         time.Time `json:"last_updated"`
	ViewCount           int       `json:"view_count"`
	ApplicationCount    int       `json:"application_count"`
	MatchCount          int       `json:"match_count"`
}

// BuildCandidateProfile creates a complete candidate profile
func BuildCandidateProfile(candidate Candidate, experiences []WorkExperience, education []Education) CandidateProfile {
	// Calculate total experience in months
	totalExp := 0
	for _, exp := range experiences {
		totalExp += exp.GetDurationInMonths()
	}
	
	// Build profile stats
	stats := ProfileStats{
		ProfileCompletion: candidate.CalculateProfileCompletion(),
		SkillCount:        len(candidate.Skills),
		ExperienceCount:   len(experiences),
		EducationCount:    len(education),
		LastUpdated:       candidate.UpdatedAt,
	}
	
	return CandidateProfile{
		Candidate:       candidate,
		WorkExperiences: experiences,
		Education:       education,
		TotalExperience: totalExp,
		ProfileStats:    stats,
	}
}
