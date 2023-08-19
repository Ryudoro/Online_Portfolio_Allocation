pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/votre-utilisateur/votre-repo.git'
            }
        }

        stage('Build & Unit Tests') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python -m unittest tests.py'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python train_model.py'
            }
        }

        stage('Predict Future') {
            steps {
                sh 'python predict_future.py'
            }
        }

        stage('Visualization') {
            steps {
                sh 'python visualize_results.py'
            }
        }

        stage('Deploy (Optional)') {
            when {
                branch 'master'
            }
            steps {
                sh 'python deploy_model.py'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'trained_model_ALOPA.h5', fingerprint: true
        }
        success {
            script {
                build(job: 'Airflow_Job_Name')
            }
        }
    }
}
