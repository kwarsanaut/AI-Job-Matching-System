package models

import (
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// MatchResult represents a job-candidate match result
type MatchResult struct {
	ID              uuid.UUID       `json:"id" db:"id"`
	JobID           uuid.UUID       `json:"job_id" db:"job_id"`
	CandidateID     uuid.UUID       `json:"candidate_id" db:"candidate_id"`
	OverallScore    float64         `json:"overall_score" db:"overall_score"`
	Confidence      float64         `json:"confidence" db:"confidence"`
	Breakdown       MatchBreakdown  `json:"breakdown" db:"breakdown"`
	MatchedSkills   pq.StringArray  `json:"matched_skills" db:"matched_skills"`
	Explanation     string          `json:"explanation" db:"explanation"`
	Recommendations []string        `json:"recommendations" db:"recommendations"`
	GeneratedAt     time.Time       `json:"generated_at" db:"generated_at"`
	
	// Joined fields
	Job             *JobPosting     `json:"job,omitempty"`
	Candidate       *Candidate      `json:"candidate,omitempty"`
}

// MatchBreakdown represents detailed score breakdown
type MatchBreakdown struct {
	SkillsScore      float64 `json:"skills_score" db:"skills_score"`
	ExperienceScore  float64 `json:"experience_score" db:"experience_score"`
	LocationScore    float64 `json:"location_score" db:"location_score"`
	SalaryScore      float64 `json:"salary_score" db:"salary_score"`
	EducationScore   float64 `json:"education_score" db:"education_score"`
	CulturalFitScore float64 `json:"cultural_fit_score" db:"cultural_fit_score"`
}

// MatchRequest represents a request to calculate matches
type MatchRequest struct {
	JobID          uuid.UUID   `json:"job_id" validate:"required"`
	CandidateIDs   []uuid.UUID `json:"candidate_ids"`
	MinScore       float64     `json:"min_score" validate:"min=0,max=100"`
	MaxResults     int         `json:"max_results" validate:"min=1,max=1000"`
	IncludeDetails bool        `json:"include_details"`
}

// BulkMatchRequest represents a bulk matching request
type BulkMatchRequest struct {
	JobIDs       []uuid.UUID `json:"job_ids"`
	CandidateIDs []uuid.UUID `json:"candidate_ids"`
	MinScore     float64     `json:"min_score" validate:"min=0,max=100"`
	MaxResults   int         `json:"max_results" validate:"min=1,max=10000"`
	Parallel     bool        `json:"parallel"`
}

// MatchSearchRequest represents match search parameters
type MatchSearchRequest struct {
	JobID       *uuid.UUID `json:"job_id" form:"job_id"`
	CandidateID *uuid.UUID `json:"candidate_id" form:"candidate_id"`
	MinScore    float64    `json:"min_score" form:"min_score" validate:"min=0,max=100"`
	MaxScore    float64    `json:"max_score" form:"max_score" validate:"min=0,max=100"`
	Limit       int        `json:"limit" form:"limit" validate:"omitempty,min=1,max=1000"`
	Offset      int        `json:"offset" form:"offset" validate:"omitempty,min=0"`
	SortBy      string     `json:"sort_by" form:"sort_by" validate:"omitempty,oneof=overall_score confidence generated_at"`
	SortOrder   string     `json:"sort_order" form:"sort_order" validate:"omitempty,oneof=asc desc"`
}

// MatchResponse represents the response for match operations
type MatchResponse struct {
	JobID           uuid.UUID     `json:"job_id"`
	CandidateID     *uuid.UUID    `json:"candidate_id,omitempty"`
	TotalMatches    int           `json:"total_matches"`
	ProcessingTime  time.Duration `json:"processing_time"`
	Matches         []MatchResult `json:"matches"`
	Metadata        MatchMetadata `json:"metadata"`
}

// MatchMetadata represents metadata about the matching process
type MatchMetadata struct {
	AlgorithmVersion string    `json:"algorithm_version"`
	ModelUsed        string    `json:"model_used"`
	ProcessedAt      time.Time `json:"processed_at"`
	FiltersApplied   []string  `json:"filters_applied"`
	ScoreDistribution map[string]int `json:"score_distribution"`
}

// MatchFilter represents filters for matching
type MatchFilter struct {
	Skills          []string `json:"skills"`
	Location        string   `json:"location"`
	ExperienceLevel string   `json:"experience_level"`
	RemoteOnly      bool     `json:"remote_only"`
	SalaryRange     *SalaryRange `json:"salary_range"`
}

// SalaryRange represents salary filtering range
type SalaryRange struct {
	Min      int    `json:"min"`
	Max      int    `json:"max"`
	Currency string `json:"currency"`
}

// MatchAnalytics represents analytics for matching performance
type MatchAnalytics struct {
	JobID               uuid.UUID `json:"job_id" db:"job_id"`
	TotalCandidates     int       `json:"total_candidates" db:"total_candidates"`
	MatchesGenerated    int       `json:"matches_generated" db:"matches_generated"`
	AvgScore            float64   `json:"avg_score" db:"avg_score"`
	HighScoreMatches    int       `json:"high_score_matches" db:"high_score_matches"` // >80
	MediumScoreMatches  int       `json:"medium_score_matches" db:"medium_score_matches"` // 60-80
	LowScoreMatches     int       `json:"low_score_matches" db:"low_score_matches"` // <60
	TopMatchedSkills    []string  `json:"top_matched_skills" db:"top_matched_skills"`
	BottleneckAreas     []string  `json:"bottleneck_areas" db:"bottleneck_areas"`
	ProcessingTimeMs    int       `json:"processing_time_ms" db:"processing_time_ms"`
	GeneratedAt         time.Time `json:"generated_at" db:"generated_at"`
}

// MatchingWeights represents the weights used in matching algorithm
type MatchingWeights struct {
	SkillsWeight     float64 `json:"skills_weight"`
	ExperienceWeight float64 `json:"experience_weight"`
	LocationWeight   float64 `json:"location_weight"`
	SalaryWeight     float64 `json:"salary_weight"`
	EducationWeight  float64 `json:"education_weight"`
	CulturalWeight   float64 `json:"cultural_weight"`
}

// DefaultMatchingWeights returns default weights for matching
func DefaultMatchingWeights() MatchingWeights {
	return MatchingWeights{
		SkillsWeight:     0.35,
		ExperienceWeight: 0.25,
		LocationWeight:   0.15,
		SalaryWeight:     0.15,
		EducationWeight:  0.05,
		CulturalWeight:   0.05,
	}
}

// GetScoreCategory returns the category of the match score
func (mr *MatchResult) GetScoreCategory() string {
	if mr.OverallScore >= 80 {
		return "excellent"
	} else if mr.OverallScore >= 60 {
		return "good"
	} else if mr.OverallScore >= 40 {
		return "fair"
	} else {
		return "poor"
	}
}

// GetConfidenceLevel returns confidence level as string
func (mr *MatchResult) GetConfidenceLevel() string {
	if mr.Confidence >= 0.8 {
		return "high"
	} else if mr.Confidence >= 0.6 {
		return "medium"
	} else {
		return "low"
	}
}

// ApplyDefaults sets default values for match result
func (mr *MatchResult) ApplyDefaults() {
	if mr.ID == uuid.Nil {
		mr.ID = uuid.New()
	}
	
	if mr.GeneratedAt.IsZero() {
		mr.GeneratedAt = time.Now()
	}
}

// Validate validates the match result
func (mr *MatchResult) Validate() error {
	if mr.JobID == uuid.Nil {
		return fmt.Errorf("job_id is required")
	}
	
	if mr.CandidateID == uuid.Nil {
		return fmt.Errorf("candidate_id is required")
	}
	
	if mr.OverallScore < 0 || mr.OverallScore > 100 {
		return fmt.Errorf("overall_score must be between 0 and 100")
	}
	
	if mr.Confidence < 0 || mr.Confidence > 1 {
		return fmt.Errorf("confidence must be between 0 and 1")
	}
	
	return nil
}

// GetTopSkills returns the top N matched skills
func (mr *MatchResult) GetTopSkills(n int) []string {
	skills := []string(mr.MatchedSkills)
	if len(skills) <= n {
		return skills
	}
	return skills[:n]
}

// MatchBatch represents a batch of matches to be processed
type MatchBatch struct {
	ID          uuid.UUID   `json:"id" db:"id"`
	JobIDs      []uuid.UUID `json:"job_ids" db:"job_ids"`
	CandidateIDs []uuid.UUID `json:"candidate_ids" db:"candidate_ids"`
	Status      string      `json:"status" db:"status"`
	Progress    float64     `json:"progress" db:"progress"`
	TotalPairs  int         `json:"total_pairs" db:"total_pairs"`
	Processed   int         `json:"processed" db:"processed"`
	Successful  int         `json:"successful" db:"successful"`
	Failed      int         `json:"failed" db:"failed"`
	StartedAt   time.Time   `json:"started_at" db:"started_at"`
	CompletedAt *time.Time  `json:"completed_at" db:"completed_at"`
	ErrorMsg    *string     `json:"error_msg" db:"error_msg"`
}

// BatchStatus constants
const (
	BatchStatusPending    = "pending"
	BatchStatusProcessing = "processing"
	BatchStatusCompleted  = "completed"
	BatchStatusFailed     = "failed"
	BatchStatusCancelled  = "cancelled"
)

// UpdateProgress updates the batch progress
func (mb *MatchBatch) UpdateProgress() {
	if mb.TotalPairs > 0 {
		mb.Progress = float64(mb.Processed) / float64(mb.TotalPairs) * 100
	}
}

// IsCompleted checks if the batch is completed
func (mb *MatchBatch) IsCompleted() bool {
	return mb.Status == BatchStatusCompleted || mb.Status == BatchStatusFailed
}

// MarkCompleted marks the batch as completed
func (mb *MatchBatch) MarkCompleted() {
	mb.Status = BatchStatusCompleted
	now := time.Now()
	mb.CompletedAt = &now
	mb.Progress = 100.0
}

// MarkFailed marks the batch as failed
func (mb *MatchBatch) MarkFailed(errorMsg string) {
	mb.Status = BatchStatusFailed
	now := time.Now()
	mb.CompletedAt = &now
	mb.ErrorMsg = &errorMsg
}

// MatchingConfig represents configuration for the matching algorithm
type MatchingConfig struct {
	Weights              MatchingWeights `json:"weights"`
	MinConfidenceThreshold float64       `json:"min_confidence_threshold"`
	MaxResultsPerJob     int             `json:"max_results_per_job"`
	EnableAIAnalysis     bool            `json:"enable_ai_analysis"`
	UseVectorSimilarity  bool            `json:"use_vector_similarity"`
	VectorWeight         float64         `json:"vector_weight"`
	CacheResults         bool            `json:"cache_results"`
	CacheDurationHours   int             `json:"cache_duration_hours"`
}

// DefaultMatchingConfig returns default matching configuration
func DefaultMatchingConfig() MatchingConfig {
	return MatchingConfig{
		Weights:                DefaultMatchingWeights(),
		MinConfidenceThreshold: 0.5,
		MaxResultsPerJob:       100,
		EnableAIAnalysis:       true,
		UseVectorSimilarity:    true,
		VectorWeight:           0.3,
		CacheResults:           true,
		CacheDurationHours:     24,
	}
}

// MatchSummary represents a summary of matching results
type MatchSummary struct {
	JobID              uuid.UUID          `json:"job_id"`
	TotalCandidates    int                `json:"total_candidates"`
	QualifiedCandidates int               `json:"qualified_candidates"`
	AvgScore           float64            `json:"avg_score"`
	BestMatch          *MatchResult       `json:"best_match,omitempty"`
	ScoreDistribution  map[string]int     `json:"score_distribution"`
	TopSkillsRequired  []string           `json:"top_skills_required"`
	SkillGaps          []string           `json:"skill_gaps"`
	Recommendations    []string           `json:"recommendations"`
	GeneratedAt        time.Time          `json:"generated_at"`
}

// GenerateMatchSummary creates a summary from match results
func GenerateMatchSummary(jobID uuid.UUID, matches []MatchResult) MatchSummary {
	summary := MatchSummary{
		JobID:           jobID,
		TotalCandidates: len(matches),
		ScoreDistribution: map[string]int{
			"excellent": 0,
			"good":      0,
			"fair":      0,
			"poor":      0,
		},
		GeneratedAt: time.Now(),
	}
	
	if len(matches) == 0 {
		return summary
	}
	
	// Calculate statistics
	totalScore := 0.0
	var bestMatch *MatchResult
	
	for i, match := range matches {
		totalScore += match.OverallScore
		
		// Track best match
		if bestMatch == nil || match.OverallScore > bestMatch.OverallScore {
			bestMatch = &matches[i]
		}
		
		// Update score distribution
		category := match.GetScoreCategory()
		summary.ScoreDistribution[category]++
		
		// Count qualified candidates (score >= 60)
		if match.OverallScore >= 60 {
			summary.QualifiedCandidates++
		}
	}
	
	summary.AvgScore = totalScore / float64(len(matches))
	summary.BestMatch = bestMatch
	
	return summary
}
