"""Environment variable validation and security checks."""
import os
import sys
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class EnvValidationError(Exception):
    """Raised when environment validation fails."""
    pass


class EnvironmentValidator:
    """Validates required environment variables and security configurations."""
    
    # Required environment variables
    REQUIRED_VARS = [
        'DATABASE_URL',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'HUGGINGFACE_API_KEY',
        'JWT_SECRET_KEY',
    ]
    
    # Insecure default values that should not be used in production
    INSECURE_DEFAULTS = {
        'JWT_SECRET_KEY': [
            'dev-secret-key',
            'your_secret_key_here',
            'your-secret-key-change-in-production-minimum-32-chars',
        ],
        'POSTGRES_PASSWORD': [
            'password',
            'postgres',
            'your_secure_password_here',
            'your_postgres_password_CHANGEME',
        ],
        'HUGGINGFACE_API_KEY': [
            'hf_xxxxxxxxxxxxx',
            'hf_your_api_key_here_CHANGEME',
        ],
    }
    
    # Minimum lengths for sensitive values
    MIN_LENGTHS = {
        'JWT_SECRET_KEY': 32,
        'POSTGRES_PASSWORD': 8,
        'HUGGINGFACE_API_KEY': 20,
    }
    
    def __init__(self, production_mode: bool = None):
        """
        Initialize validator.
        
        Args:
            production_mode: If True, enforce strict validation. If None, auto-detect.
        """
        if production_mode is None:
            # Auto-detect from environment or BUILD_TARGET
            build_target = os.getenv('BUILD_TARGET', 'development')
            production_mode = build_target == 'production'
        
        self.production_mode = production_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all environment variables.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Check required variables exist
        self._check_required_vars()
        
        # Check for insecure defaults
        self._check_insecure_defaults()
        
        # Check minimum lengths
        self._check_min_lengths()
        
        # Validate specific formats
        self._validate_database_url()
        self._validate_api_keys()
        
        # Check file permissions (if .env exists)
        self._check_env_file_permissions()
        
        is_valid = len(self.errors) == 0
        
        return is_valid, self.errors, self.warnings
    
    def _check_required_vars(self):
        """Check that all required environment variables are set."""
        for var in self.REQUIRED_VARS:
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing required environment variable: {var}")
            elif not value.strip():
                self.errors.append(f"Environment variable {var} is empty")
    
    def _check_insecure_defaults(self):
        """Check for insecure default values."""
        for var, insecure_values in self.INSECURE_DEFAULTS.items():
            value = os.getenv(var, '').strip()
            if value in insecure_values:
                if self.production_mode:
                    self.errors.append(
                        f"{var} is using an insecure default value. "
                        f"Generate a secure value before deploying."
                    )
                else:
                    self.warnings.append(
                        f"{var} is using a default value. "
                        f"Consider setting a secure value for production."
                    )
    
    def _check_min_lengths(self):
        """Check that sensitive values meet minimum length requirements."""
        for var, min_length in self.MIN_LENGTHS.items():
            value = os.getenv(var, '')
            if value and len(value) < min_length:
                if self.production_mode:
                    self.errors.append(
                        f"{var} is too short (minimum {min_length} characters). "
                        f"Use a longer, more secure value."
                    )
                else:
                    self.warnings.append(
                        f"{var} is shorter than recommended ({min_length} characters)"
                    )
    
    def _validate_database_url(self):
        """Validate DATABASE_URL format."""
        db_url = os.getenv('DATABASE_URL', '')
        if db_url and not db_url.startswith('postgresql://'):
            self.errors.append(
                "DATABASE_URL must start with 'postgresql://' for PostgreSQL connections"
            )
        
        # Check if password is in URL and exposed
        if '@' in db_url:
            parts = db_url.split('@')
            if len(parts) == 2 and ':' in parts[0]:
                password = parts[0].split(':')[-1]
                if password in self.INSECURE_DEFAULTS.get('POSTGRES_PASSWORD', []):
                    self.warnings.append(
                        "DATABASE_URL contains an insecure password"
                    )
    
    def _validate_api_keys(self):
        """Validate API key formats."""
        hf_key = os.getenv('HUGGINGFACE_API_KEY', '')
        if hf_key and not hf_key.startswith('hf_'):
            self.warnings.append(
                "HUGGINGFACE_API_KEY should start with 'hf_' prefix"
            )
    
    def _check_env_file_permissions(self):
        """Check .env file permissions for security."""
        env_file = '.env'
        if os.path.exists(env_file):
            import stat
            file_stats = os.stat(env_file)
            file_mode = file_stats.st_mode
            
            # Check if file is world-readable (dangerous)
            if file_mode & stat.S_IROTH:
                self.warnings.append(
                    ".env file is world-readable. Run: chmod 600 .env"
                )
            
            # Check if file is group-readable
            if file_mode & stat.S_IRGRP:
                self.warnings.append(
                    ".env file is group-readable. Run: chmod 600 .env"
                )
    
    def validate_or_exit(self):
        """
        Validate environment and exit with error if validation fails.
        
        Use this at application startup to ensure environment is properly configured.
        """
        is_valid, errors, warnings = self.validate()
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")
        
        # Log and exit on errors
        if not is_valid:
            logger.error("❌ Environment validation failed:")
            for error in errors:
                logger.error(f"   • {error}")
            
            logger.error("\nPlease fix the above errors before starting the application.")
            logger.error("See .env.example for configuration template.")
            sys.exit(1)
        
        logger.info("✅ Environment validation passed")
        
        if warnings:
            logger.info(f"⚠️  {len(warnings)} warnings found (see logs above)")


def validate_environment(production_mode: bool = None) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate environment.
    
    Args:
        production_mode: If True, enforce strict validation
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = EnvironmentValidator(production_mode=production_mode)
    return validator.validate()


def ensure_environment():
    """
    Validate environment and exit if invalid.
    
    Use this at application startup.
    """
    validator = EnvironmentValidator()
    validator.validate_or_exit()


if __name__ == '__main__':
    # Test validation
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    validator = EnvironmentValidator()
    is_valid, errors, warnings = validator.validate()
    
    logger.info(f"\n{'='*60}")
    logger.info("Environment Validation Report")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Status: {'✅ VALID' if is_valid else '❌ INVALID'}\n")
    
    if errors:
        logger.error(f"❌ Errors ({len(errors)}):")
        for error in errors:
            logger.error(f"   • {error}")
    
    if warnings:
        logger.warning(f"⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            logger.warning(f"   • {warning}")
    
    if is_valid and not warnings:
        logger.info("✨ All checks passed!")
    
    logger.info(f"{'='*60}\n")
    
    sys.exit(0 if is_valid else 1)
