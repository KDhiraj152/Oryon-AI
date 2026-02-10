"""Add multi-tenancy support

Revision ID: 016_add_multi_tenancy
Revises: 015_add_learning_recommendations
Create Date: 2024-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '016_add_multi_tenancy'
down_revision = '015_add_learning_recommendations'
branch_labels = None
depends_on = None

# Constants for server defaults
NOW_DEFAULT = 'now()'


def upgrade():
    # Organizations table
    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('domain', sa.String(255), nullable=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('settings', postgresql.JSONB, nullable=True),
        sa.Column('subscription_tier', sa.String(50), default='free'),
        sa.Column('max_users', sa.Integer, default=10),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text(NOW_DEFAULT), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text(NOW_DEFAULT), nullable=False)
    )

    op.create_index('idx_organizations_slug', 'organizations', ['slug'])
    op.create_index('idx_organizations_domain', 'organizations', ['domain'])

    # Add organization_id to users table
    op.add_column('users', sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_users_organization',
        'users', 'organizations',
        ['organization_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('idx_users_organization_id', 'users', ['organization_id'])

    # Add organization_id to all main tables (check if tables exist first)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    tables_to_update = [
        'processed_content',
    ]

    # Only update tables that actually exist
    tables_to_update = [t for t in tables_to_update if t in existing_tables]

    for table in tables_to_update:
        # Check if column already exists
        columns = [col['name'] for col in inspector.get_columns(table)]
        if 'organization_id' not in columns:
            op.add_column(table, sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True))
            op.create_foreign_key(
                f'fk_{table}_organization',
                table, 'organizations',
                ['organization_id'], ['id'],
                ondelete='CASCADE'
            )
            op.create_index(f'idx_{table}_organization_id', table, ['organization_id'])

    # Enable Row Level Security (RLS)
    conn = op.get_bind()

    # Enable RLS on all tables
    for table in ['organizations', 'users'] + tables_to_update:
        conn.execute(sa.text(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;'))

    # Create RLS policies
    # Policy 1: Users can only see data from their organization
    for table in tables_to_update:
        conn.execute(sa.text(f"""
            CREATE POLICY {table}_isolation_policy ON {table}
            USING (organization_id = current_setting('app.current_organization_id', true)::uuid);
        """))

    # Policy 2: Users can see their own user record
    conn.execute(sa.text("""
        CREATE POLICY users_own_record_policy ON users
        USING (id = current_setting('app.current_user_id', true)::uuid);
    """))

    # Policy 3: Users can see users in their organization
    conn.execute(sa.text("""
        CREATE POLICY users_same_org_policy ON users
        USING (organization_id = current_setting('app.current_organization_id', true)::uuid);
    """))

    # Organization members junction table
    op.create_table(
        'organization_members',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(50), default='member'),
        sa.Column('invited_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.text(NOW_DEFAULT), nullable=False),

        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['invited_by'], ['users.id'], ondelete='SET NULL'),

        sa.UniqueConstraint('organization_id', 'user_id', name='uq_org_user_membership')
    )

    op.create_index('idx_organization_members_org_id', 'organization_members', ['organization_id'])
    op.create_index('idx_organization_members_user_id', 'organization_members', ['user_id'])


def downgrade():
    # Disable RLS and drop policies
    conn = op.get_bind()

    tables = [
        'organizations', 'users', 'processed_content',
        'documents', 'translation_reviews', 'user_performance', 'learning_paths'
    ]

    for table in tables:
        # Drop policies
        conn.execute(sa.text(f"DROP POLICY IF EXISTS {table}_isolation_policy ON {table};"))
        # Disable RLS
        conn.execute(sa.text(f'ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;'))

    conn.execute(sa.text("DROP POLICY IF EXISTS users_own_record_policy ON users;"))
    conn.execute(sa.text("DROP POLICY IF EXISTS users_same_org_policy ON users;"))

    # Drop tables
    op.drop_table('organization_members')

    # Remove organization_id columns
    tables_to_update = [
        'learning_paths', 'user_performance', 'translation_reviews',
        'documents', 'processed_content', 'users'
    ]

    for table in tables_to_update:
        op.drop_constraint(f'fk_{table}_organization', table, type_='foreignkey')
        op.drop_index(f'idx_{table}_organization_id', table)
        op.drop_column(table, 'organization_id')

    op.drop_table('organizations')
