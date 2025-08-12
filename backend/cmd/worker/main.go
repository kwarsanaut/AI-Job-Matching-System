package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"job-matching-system/internal/config"
	"job-matching-system/internal/database"
	"job-matching-system/internal/services"
	"job-matching-system/internal/worker"
)

func main() {
	log.Println("🔄 Starting background worker...")
	
	// Load configuration
	cfg := config.Load()
	
	// Initialize database
	db, err := database.NewConnection(cfg.DatabaseURL)
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	defer db.Close()
	
	// Initialize services
	services := services.NewServices(db, cfg)
	
	// Initialize worker pools
	workerManager := worker.NewManager(services, cfg)
	
	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// Start worker pools
	var wg sync.WaitGroup
	
	// Embedding processing worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerManager.StartEmbeddingWorker(ctx)
	}()
	
	// Matching computation worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerManager.StartMatchingWorker(ctx)
	}()
	
	// Email notification worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerManager.StartEmailWorker(ctx)
	}()
	
	// Analytics processing worker
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerManager.StartAnalyticsWorker(ctx)
	}()
	
	// Cleanup worker (runs periodically)
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerManager.StartCleanupWorker(ctx)
	}()
	
	log.Println("✅ All workers started successfully")
	
	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	
	log.Println("🛑 Shutting down workers...")
	
	// Cancel context to stop all workers
	cancel()
	
	// Wait for all workers to finish with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		log.Println("✅ All workers stopped gracefully")
	case <-time.After(30 * time.Second):
		log.Println("⚠️ Workers shutdown timeout, forcing exit")
	}
	
	// Shutdown services
	services.Shutdown()
	
	log.Println("✅ Worker process exited")
}
