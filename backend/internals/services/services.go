type Services struct {
    JobService        JobService
    CandidateService  CandidateService
    MatchingService   MatchingService
    EmailService      EmailService
    AIService         AIService
}

func NewServices(db *sql.DB, cfg *config.Config) *Services
