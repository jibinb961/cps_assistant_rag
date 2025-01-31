create or replace function match_site_pages (
  query_embedding vector(768),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  search_mode varchar default 'general'
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
declare
  top_program_title varchar;
begin
 if search_mode = 'general' then
   -- For general search, retrieve top 5 most relevant chunks across all programs
   return query
   select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
   from site_pages
   where metadata @> filter
   order by similarity desc
   limit 5;
 else
  -- Step 1: Find the most relevant program based on the title and metadata
  select title into top_program_title
  from site_pages
  where metadata @> filter
  order by 1 - (site_pages.embedding <=> query_embedding) desc
  limit 1;

  -- Step 2: Retrieve all three chunks for the most relevant program
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where title = top_program_title and metadata @> filter
  order by chunk_number
  limit 3;
  end if;
end;
$$;