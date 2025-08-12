type MatchRepository interface {
    Store(ctx context.Context, match *models.MatchResult) error
    GetMatches(ctx context.Context, req models.MatchSearchRequest) ([]models.MatchResult, error)
    GetJobMatches(ctx context.Context, jobID uuid.UUID) ([]models.MatchResult, error)
    GetCandidateMatches(ctx context.Context, candidateID uuid.UUID) ([]models.MatchResult, error)
}
