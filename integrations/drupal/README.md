# Drupal Integration for SEED

To integrate SEED Semantic Search with Drupal, use the following pattern:

## Prerequisites

1. A Drupal site with the `Search API` and `Search API Solr` modules configured.
2. The SEED gateway running and accessible from the Drupal server.

## Integration Steps

### 1. Configure SEED API Key
Ensure your SEED gateway has an API key set in `config.yaml` or as an environment variable `SEED_API_KEY`.

### 2. Custom Module Request
You can create a custom block or controller to call the SEED gateway:

```php
$client = \Drupal::httpClient();
$response = $client->post('http://seed-gateway:8000/v1/search/semantic', [
  'headers' => [
    'Authorization' => 'Bearer ' . $api_key,
    'Content-Type' => 'application/json',
  ],
  'json' => [
    'query' => $user_query,
    'facets' => ['field_topic:Energy'],
  ],
]);

$data = json_decode($response->getBody()->getContents(), TRUE);
$answer = $data['answer'];
$results = $data['results'];
```

## Mapping Fields
The SEED gateway uses `search_api_fulltext` as the default query field, which matches common Drupal Search API conventions. Ensure your Solr core matches the `solr.url` in `config.yaml`.
",filePath:"/Users/mmh/Documents/data_science_kdf/integrations/drupal/README.md"}end"filePath: