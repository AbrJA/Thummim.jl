url = "https://openbible.com/textfiles/akjv.txt"

sentences = read(download(url)) |> String |> x -> split(x, "\n")
filter!(x-> occursin(r"\d+:\d+", x), sentences)

using UUIDs
using Random

rng = Xoshiro(123);
namespace = uuid4(rng)

struct Verse
    id::UUID
    book::String
    chapter::Int
    verse::Int
    text::String
end

bible = Vector{Verse}()
# bible = Dict{Tuple{String,Int,Int}, String}()

for sentence in sentences
    try
        ref, text = split(sentence, "\t")
        book, chapter_verse = split(ref, r"\s+(?=\d+:\d+)")
        chapter, verse = split(chapter_verse, ":")
        # bible[(book, parse(Int, chapter), parse(Int, verse))] = text
        push!(bible, Verse(uuid5(namespace, string(text)), book, parse(Int, chapter), parse(Int, verse), text))
    catch e
        error("Failed to parse sentence: $sentence")
    end
end

# findfirst(x -> occursin("1 Samuel 1:1", x), sentences)
# sentence = sentences[7214]
# bible[("Ecclesiastes", 1, 2)]

using HTTP
using JSON3
using LinearAlgebra
using PromptingTools

const PT = PromptingTools
schema = PT.OllamaSchema()

url = "http://localhost:6333"
collection = "/collections/bible"

# HTTP.delete(string(url, collection))
body = Dict(:vectors => Dict(:size => 384, :distance => "Dot"))
response = HTTP.put(string(url, collection), body = JSON3.write(body))

for verse in bible
    answer = aiembed(schema, verse.text, model = "all-minilm")
    point = Dict(:id => verse.id,
                 :payload => Dict(:book => verse.book,
                                  :chapter => verse.chapter,
                                  :verse => verse.verse,
                                  :text => verse.text),
                  :vector => normalize(answer.content))
    body = Dict(:points => [point])
    response = HTTP.put(string(url, collection, "/points"), body = JSON3.write(body))
end

struct Search
    vector::AbstractVector
    limit::Int
    with_payload::Bool
    with_vector::Bool
    filter::Dict{Symbol, Any}

    function Search2(vector::AbstractVector,
                    limit::Int,
                    with_payload::Bool,
                    with_vector::Bool;
                    kwargs...)
        allowed_keys = [:must, :should, :must_not]
        for key in keys(kwargs)
            if key ∉ allowed_keys
                throw(ArgumentError("Invalid key: $key"))
            end
        end
        new(vector, limit, with_payload, with_vector, kwargs)
    end
end

function search(params::Search)
    try
        body = JSON3.write(params)
        response = HTTP.post(string(url, collection, "/points/search"), body = body)
        JSON3.read(String(response.body))
    catch error
        @error "Failed to search points. Error: $error"
    end
end

answer = aiembed(schema, "Teach the child on the way", model = "all-minilm")
limit = 3
with_payload = true
with_vector = false
filter = Dict(:key => "book", :match => Dict(:value => "Proverbs"))
params = Search(answer.content, limit, with_payload, with_vector, must = filter)

response = search(params)
response[:result]
