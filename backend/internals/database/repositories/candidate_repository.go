type CandidateRepository interface {
    Create(ctx context.Context, candidate *models.Candidate) error
    GetByID(ctx context.Context, id uuid.UUID) (*models.Candidate, error)
    GetByEmail(ctx context.Context, email string) (*models.Candidate, error)
    Update(ctx context.Context, candidate *models.Candidate) error
    Search(ctx context.Context, req models.CandidateSearchRequest) ([]models.Candidate, int, error)
}
