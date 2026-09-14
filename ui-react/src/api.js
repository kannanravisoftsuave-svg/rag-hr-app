import axios from 'axios'

const client = axios.create({ baseURL: '/api', timeout: 300000 })

export const getHealth = () => client.get('/health')
export const getCollections = () => client.get('/collections')
export const getDocuments = (collection) => client.get('/documents', { params: { collection } })
export const getChunks = (source_file, collection) =>
  client.get('/chunks', { params: { source_file, collection } })

export const uploadDocument = (file, collection, chunking_strategy) => {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection)
  form.append('chunking_strategy', chunking_strategy)
  return client.post('/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export const queryDocuments = (payload) => client.post('/query', payload)
