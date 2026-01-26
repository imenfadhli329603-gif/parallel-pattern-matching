#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>   // Per std::accumulate e std::inner_product
#include <algorithm> // Per std::sort
#include <omp.h>

// Funzione per leggere la colonna 'temperature_c' dal file CSV (Indice 3)
std::vector<double> read_csv(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        exit(1); // Termina il programma se il file non si trova
    }
    std::string line, val;
    
    std::getline(file, line); // Salta l'intestazione
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;
        while (std::getline(ss, val, ',')) {
            row.push_back(val);
        }
        if (row.size() > 3 && !row[3].empty()) {
            try {
                data.push_back(std::stod(row[3])); // Colonna temperature_c
            } catch (const std::invalid_argument& e) {
                // Ignora le righe con valori non numerici
            }
        }
    }
    return data;
}

// Implementazione Parallela con OpenMP e Vettorizzazione SIMD
double find_pattern_parallel(const std::vector<double>& series, const std::vector<double>& pattern, int& best_index) {
    int n = series.size();
    int m = pattern.size();
    double global_min_sad = 1e30; // Un valore iniziale molto grande
    int global_best_index = -1;

    #pragma omp parallel
    {
        double local_min_sad = 1e30;
        int local_best_index = -1;

        #pragma omp for nowait
        for (int i = 0; i <= n - m; i++) {
            double current_sad = 0.0;
            
            #pragma omp simd reduction(+:current_sad)
            for (int j = 0; j < m; j++) {
                current_sad += std::abs(series[i + j] - pattern[j]);
            }

            if (current_sad < local_min_sad) {
                local_min_sad = current_sad;
                local_best_index = i;
            }
        }

        #pragma omp critical
        {
            if (local_min_sad < global_min_sad) {
                global_min_sad = local_min_sad;
                global_best_index = local_best_index;
            }
        }
    }
    best_index = global_best_index;
    return global_min_sad;
}

// Funzione per calcolare la media
double calculate_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// Funzione per calcolare la deviazione standard
double calculate_stddev(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return std::sqrt(sq_sum / v.size() - mean * mean);
}


int main() {
    // ========================================================================
    // PASSO 1: PREPARAZIONE DEL DATASET (Regola dei 10 secondi)
    // ========================================================================
    std::cout << "--- Passo 1: Preparazione del Dataset ---" << std::endl;
    std::vector<double> base_data = read_csv("iot_weather_sensor_data.csv");
    if (base_data.empty()) {
        std::cerr << "Nessun dato caricato. Il programma termina." << std::endl;
        return 1;
    }
    std::vector<double> query(base_data.begin(), base_data.begin() + std::min((size_t)100, base_data.size()));

    std::vector<double> large_series;
    double sequential_time = 0.0;
    int temp_idx;

    omp_set_num_threads(1); // Forza l'uso di 1 solo thread per questo test
    while (sequential_time < 10.0) {
        large_series.insert(large_series.end(), base_data.begin(), base_data.end());
        
        double start = omp_get_wtime();
        find_pattern_parallel(large_series, query, temp_idx);
        double end = omp_get_wtime();
        sequential_time = end - start;
        
        std::cout << "Dimensione attuale dataset: " << large_series.size() 
                  << ", Tempo sequenziale misurato: " << sequential_time << "s" << std::endl;
    }
    std::cout << "Dataset finale creato. Dimensione: " << large_series.size() 
              << ", Tempo sequenziale di riferimento: " << sequential_time << "s\n" << std::endl;

    // ========================================================================
    // PASSO 2, 3, 4: RACCOLTA SISTEMATICA DEI DATI
    // ========================================================================
    std::cout << "--- Passo 2, 3, 4: Raccolta Dati di Performance ---" << std::endl;
    
    // Configurazioni da testare
    std::vector<int> threads_to_test = {1, 2, 4, 8, 12, 16, 24, 32, 64};
    int num_runs = 10; // Numero di ripetizioni per ogni test
    double base_sequential_time = 0.0;

    // Stampa l'intestazione per la tabella CSV
    std::cout << "\nThreads,TempoMedio(s),DeviazioneStandard(s),Speedup" << std::endl;

    for (int p : threads_to_test) {
        omp_set_num_threads(p);
        
        std::vector<double> timings;

        // Warm-up
        find_pattern_parallel(large_series, query, temp_idx);

        // Esecuzioni ripetute
        for (int i = 0; i < num_runs; ++i) {
            double start = omp_get_wtime();
            find_pattern_parallel(large_series, query, temp_idx);
            double end = omp_get_wtime();
            timings.push_back(end - start);
        }

        // Calcolo statistiche
        double mean_time = calculate_mean(timings);
        double std_dev = calculate_stddev(timings, mean_time);

        if (p == 1) {
            base_sequential_time = mean_time;
        }

        double speedup = (base_sequential_time > 0) ? base_sequential_time / mean_time : 0.0;

        // Stampa i risultati in formato CSV
        std::cout << p << "," << mean_time << "," << std_dev << "," << speedup << std::endl;
    }

    // ========================================================================
    // PASSO 5: DOCUMENTAZIONE AMBIENTE
    // ========================================================================
    std::cout << "\n--- Passo 5: Documentazione Ambiente ---" << std::endl;
    std::cout << "Ricorda di annotare le specifiche del tuo PC nel report:" << std::endl;
    std::cout << " - CPU: Modello, Numero di core fisici" << std::endl;
    std::cout << " - RAM: Quantita' totale" << std::endl;
    std::cout << " - OS: Sistema Operativo (es. Windows 11, Ubuntu 22.04)" << std::endl;
    std::cout << " - Compilatore: Versione (es. g++ 11.2.0, MSVC v19.30)" << std::endl;
    std::cout << " - Flag di Compilazione: (es. -O3 -fopenmp)" << std::endl;

    return 0;
}
