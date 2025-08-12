type JobService interface {
    CreateJob(ctx context.Context, req models.JobCreateRequest) (*models.JobPosting, error)
    GetJob(ctx context.Context, id uuid.UUID) (*models.JobPosting, error)
    UpdateJob(ctx context.Context, id uuid.UUID, req models.JobUpdateRequest) (*models.JobPosting, error)
    DeleteJob(ctx context.Context, id uuid.UUID) error
    SearchJobs(ctx context.Context, req models.JobSearchRequest) ([]models.JobPosting, int, error)
    GetJobAnalytics(ctx context.Context, jobID uuid.UUID) (*models.JobAnalytics, error)
}
