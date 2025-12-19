-- ===========================================================
-- EXTENSIONS
-- ===========================================================

create extension if not exists "pgcrypto" with schema public;
create extension if not exists "uuid-ossp" with schema public;
create extension if not exists "vector" with schema public;

-- ===========================================================
-- TABLE: expert_documents
-- ===========================================================

create table if not exists public.expert_documents (
  id uuid primary key default gen_random_uuid(),
  expert_id uuid not null,
  file_name text,
  storage_path text,
  public_url text,
  created_at timestamptz default now()
);

alter table public.expert_documents enable row level security;

create policy experts_can_read_own_rows
on public.expert_documents
for select
to public
using (auth.uid() = expert_id);

create policy experts_can_insert_own_rows
on public.expert_documents
for insert
to public
with check (auth.uid() = expert_id);

create policy experts_can_delete_own_rows
on public.expert_documents
for delete
to public
using (auth.uid() = expert_id);

-- ===========================================================
-- TABLE: expert_ratings
-- ===========================================================

create table if not exists public.expert_ratings (
  id uuid primary key default gen_random_uuid(),
  expert_id text not null,
  user_id uuid not null,
  rating integer not null check (rating between 1 and 5),
  satisfied boolean,
  reasons text[] default '{}'::text[],
  question_count_at_rating integer,
  created_at timestamptz default now()
);

create index if not exists expert_ratings_user_id_idx
  on public.expert_ratings (user_id);

create index if not exists expert_ratings_expert_id_idx
  on public.expert_ratings (expert_id);

create unique index if not exists expert_ratings_unique_user_expert
  on public.expert_ratings (expert_id, user_id);

alter table public.expert_ratings enable row level security;

create policy ratings_are_public
on public.expert_ratings
for select
to public
using (true);

create policy user_can_insert_own_rating
on public.expert_ratings
for insert
to public
with check (auth.uid() = user_id);

create policy user_can_update_own_rating
on public.expert_ratings
for update
to public
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

-- ===========================================================
-- TABLE: user_favorite_experts
-- ===========================================================

create table if not exists public.user_favorite_experts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  expert_id text not null,
  created_at timestamptz default now()
);

create index if not exists user_favorites_user_id_idx
  on public.user_favorite_experts (user_id);

create index if not exists user_favorites_expert_id_idx
  on public.user_favorite_experts (expert_id);

create unique index if not exists user_favorites_unique
  on public.user_favorite_experts (user_id, expert_id);

alter table public.user_favorite_experts enable row level security;

create policy user_can_select_own_favourites
on public.user_favorite_experts
for select
to public
using (auth.uid() = user_id);

create policy user_can_insert_own_favourites
on public.user_favorite_experts
for insert
to public
with check (auth.uid() = user_id);

create policy user_can_delete_own_favourites
on public.user_favorite_experts
for delete
to public
using (auth.uid() = user_id);

-- ===========================================================
-- TABLE: expert_profiles (RLS disabled)
-- ===========================================================

create table if not exists public.expert_profiles (
  id uuid primary key default gen_random_uuid(),
  expert_id text not null,
  short_description text,
  cv_file_name text,
  cv_public_url text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  owner_id uuid,
  constraint expert_profiles_expert_id_key unique (expert_id),
  constraint expert_profiles_owner_id_fkey
    foreign key (owner_id) references auth.users (id)
) tablespace pg_default;

alter table public.expert_profiles disable row level security;

create trigger set_updated_at
before update on public.expert_profiles
for each row
execute function set_updated_at();

-- ===========================================================
-- TABLE: chat_messages
-- ===========================================================

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  expert_id text not null,
  user_id uuid,
  role text not null,
  content text not null,
  created_at timestamptz default now(),
  constraint chat_messages_user_id_fkey
    foreign key (user_id) references auth.users (id) on delete cascade,
  constraint chat_messages_role_check
    check (role in ('user', 'assistant'))
) tablespace pg_default;

create index if not exists idx_chat_messages_expert_user
  on public.chat_messages (expert_id, user_id, created_at);

alter table public.chat_messages enable row level security;

create policy users_can_insert_own_messages
on public.chat_messages
for insert
to public
with check (auth.uid() = user_id);

create policy users_can_view_own_messages
on public.chat_messages
for select
to public
using (auth.uid() = user_id);

-- ===========================================================
-- TABLE: documents
-- ===========================================================

create table if not exists public.documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector
) tablespace pg_default;

alter table public.documents enable row level security;

create policy service_role_can_select_documents
on public.documents
for select
to service_role
using (true);

create policy service_role_can_insert_documents
on public.documents
for insert
to service_role
with check (true);

create policy service_role_can_update_documents
on public.documents
for update
to service_role
using (true)
with check (true);

create policy service_role_can_delete_documents
on public.documents
for delete
to service_role
using (true);

-- ===========================================================
-- TABLE: n8n_chat_histories
-- ===========================================================

create table if not exists public.n8n_chat_histories (
  id serial primary key,
  session_id uuid not null,
  message jsonb not null
) tablespace pg_default;

alter table public.n8n_chat_histories enable row level security;

create policy service_role_can_select_n8n_chat_histories
on public.n8n_chat_histories
for select
to service_role
using (true);

create policy service_role_can_insert_n8n_chat_histories
on public.n8n_chat_histories
for insert
to service_role
with check (true);

create policy service_role_can_update_n8n_chat_histories
on public.n8n_chat_histories
for update
to service_role
using (true)
with check (true);

create policy service_role_can_delete_n8n_chat_histories
on public.n8n_chat_histories
for delete
to service_role
using (true);

-- ===========================================================
-- FUNCTION: match_documents
-- ===========================================================

create or replace function public.match_documents (
  query_embedding vector(4096),
  match_count integer default 5,
  filter jsonb default '{}'::jsonb
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity double precision
)
language plpgsql
as $$
begin
  return query
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  where filter = '{}'::jsonb
     or d.metadata @> filter
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;
