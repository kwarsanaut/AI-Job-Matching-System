type EmailService interface {
    SendWelcomeEmail(ctx context.Context, candidate *models.Candidate) error
    SendJobMatchNotification(ctx context.Context, match *models.MatchResult) error
    SendApplicationConfirmation(ctx context.Context, application *models.Application) error
}
