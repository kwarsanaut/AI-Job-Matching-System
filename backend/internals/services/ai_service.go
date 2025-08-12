type AIService interface {
    GenerateEmbedding(ctx context.Context, text string) ([]float32, error)
    CalculateMatch(ctx context.Context, jobID, candidateID uuid.UUID) (*models.MatchResult, error)
    FindCandidatesForJob(ctx context.Context, jobID uuid.UUID, limit int) ([]models.MatchResult, error)
    FindJobsForCandidate(ctx context.Context, candidateID uuid.UUID, limit int) ([]models.MatchResult, error)
    AnalyzeJobRequirements(ctx context.Context, jobDesc string) (map[string]interface{}, error)
}
