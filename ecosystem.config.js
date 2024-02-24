module.exports = {
    apps: [{
        name: 'flask-app',
        script: 'app.py',
        interpreter: 'python3',
        watch: true,
        ignore_watch: ["node_modules", "logs"],
        env: {
            PORT: 80,
            FLASK_ENV: 'development'
        },
        env_production: {
            PORT: 80,
            FLASK_ENV: 'production'
        }
    }]
};
