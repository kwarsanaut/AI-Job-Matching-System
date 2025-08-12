type JobRepository interface {
    Create(ctx context.Context, job *models.JobPosting) error
    GetByID(ctx context.Context, id uuid.UUID) (*models.JobPosting, error)
    Update(ctx context.Context, job *models.JobPosting) error
    Delete(ctx context.Context, id uuid.UUID) error
    Search(ctx context.Context, req models.JobSearchRequest) ([]models.JobPosting, int, error)
    GetByCompany(ctx context.Context, companyID uuid.UUID) ([]models.JobPosting, error)
}
