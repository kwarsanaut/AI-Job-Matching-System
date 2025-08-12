func NewConnection(databaseURL string) (*sql.DB, error)
func NewPostgresPool(cfg PoolConfig) (*pgxpool.Pool, error)
func HealthCheck(db *sql.DB) error
