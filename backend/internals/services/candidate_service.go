type CandidateService interface {
    CreateCandidate(ctx context.Context, req models.CandidateCreateRequest) (*models.Candidate, error)
    GetCandidate(ctx context.Context, id uuid.UUID) (*models.Candidate, error)
    UpdateCandidate(ctx context.Context, id uuid.UUID, req models.CandidateUpdateRequest) (*models.Candidate, error)
    SearchCandidates(ctx context.Context, req models.CandidateSearchRequest) ([]models.Candidate, int, error)
    AddWorkExperience(ctx context.Context, req models.WorkExperienceCreateRequest) (*models.WorkExperience, error)
    AddEducation(ctx context.Context, req models.EducationCreateRequest) (*models.Education, error)
}
