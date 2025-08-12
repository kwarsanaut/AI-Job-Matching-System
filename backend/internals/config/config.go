type Config struct {
    Environment     string
    Port           string
    DatabaseURL    string
    RedisURL       string
    QdrantURL      string
    OpenAIAPIKey   string
    JWTSecret      string
    RateLimit      int
    EnableMetrics  bool
    ReadTimeout    int
    WriteTimeout   int
    IdleTimeout    int
}

func Load() *Config
func (c *Config) Validate() error
