/*  Adapted from Model.cpp in OWL repository by Ingo Wald.
    https://github.com/owl-project/
*/

#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace std{
    inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

/*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
int addVertex(TriangleMesh *mesh,
              tinyobj::attrib_t &attributes,
              const tinyobj::index_t &idx,
              std::map<tinyobj::index_t,int> &knownVertices)
{
    if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    const owl::vec3f *vertex_array   = (const owl::vec3f*)attributes.vertices.data();
    const owl::vec3f *normal_array   = (const owl::vec3f*)attributes.normals.data();
    const owl::vec2f *texcoord_array = (const owl::vec2f*)attributes.texcoords.data();

    int newID = (int)mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
        while (mesh->normal.size() < mesh->vertex.size())
            mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
        while (mesh->texcoord.size() < mesh->vertex.size())
            mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    return newID;
}

/*! load a texture (if not already loaded), and return its ID in the
    model's textures[] vector. Textures that could not get loaded
    return -1 */
int loadTexture(Model *model,
                std::map<std::string,int> &knownTextures,
                const std::string &inFileName,
                const std::string &modelPath)
{
    if (inFileName == "")
        return -1;  

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto &c : fileName)
        if (c == '\\') c = '/';
    if (fileName.empty() || fileName[0] != '/')
        fileName = modelPath+"/"+fileName;

    owl::vec2i res;
    int   comp;
    unsigned char* image = stbi_load(fileName.c_str(),
                                     &res.x, &res.y, &comp, STBI_rgb_alpha);
    int textureID = -1;
    if (image) {
        textureID = (int)model->textures.size();
        Texture *texture = new Texture;
        texture->resolution = res;
        texture->pixel      = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
        mirrored along the y axis - mirror them here */
        for (int y=0;y<res.y/2;y++) {
            uint32_t *line_y = texture->pixel + y * res.x;
            uint32_t *mirrored_y = texture->pixel + (res.y-1-y) * res.x;
            for (int x=0;x<res.x;x++) {
                std::swap(line_y[x],mirrored_y[x]);
            }
        }

        model->textures.push_back(texture);
    }
    else {
        std::cout << OWL_TERMINAL_RED
                  << "Could not load texture from " << fileName << "!"
                  << OWL_TERMINAL_DEFAULT << std::endl;
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

static std::string parentDir(const std::string &path)
{
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash + 1);
}

Model *loadOBJ(const std::string &objFile,
               const std::string &materialFile,
               const std::string &textureFile)
{
    Model *model = new Model;

    const std::string modelDir = parentDir(objFile);
    const std::string materialDir = materialFile.empty() ? modelDir : parentDir(materialFile);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::vector<tinyobj::material_t> overrideMaterials;
    std::string err = "";

    bool readOK = tinyobj::LoadObj( &attributes,
                                    &shapes,
                                    &materials,
                                    &err,
                                    &err,
                                    objFile.c_str(),
                                    modelDir.c_str(),
                                    /* triangulate */true);
    if (!materialFile.empty()) {
        std::ifstream materialStream(materialFile);
        if (!materialStream)
            throw std::runtime_error("Could not read material file from "+materialFile);

        std::map<std::string, int> materialMap;
        std::string materialWarning;
        std::string materialError;
        tinyobj::LoadMtl(&materialMap,
                         &overrideMaterials,
                         &materialStream,
                         &materialWarning,
                         &materialError);
        if (!materialWarning.empty())
            std::cout << materialWarning << std::endl;
        if (!materialError.empty())
            std::cerr << materialError << std::endl;
        if (!overrideMaterials.empty())
            materials = overrideMaterials;
    }
    if (!readOK)
        throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);

    /* We only render primitives - no need to render material/texture. */
    // if (materials.empty())
    //     throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
        tinyobj::shape_t &shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        std::map<tinyobj::index_t,int> knownVertices;
        std::map<std::string,int>      knownTextures;

        for (int materialID : materialIDs) {
            TriangleMesh *mesh = new TriangleMesh;

            for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                if (shape.mesh.num_face_vertices[faceID] != 3)
                    throw std::runtime_error("not properly tessellated");
                tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
                tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
                tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];

                owl::vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                               addVertex(mesh, attributes, idx1, knownVertices),
                               addVertex(mesh, attributes, idx2, knownVertices));
                mesh->index.push_back(idx);
                const int effectiveMaterialID = (materialID < 0 && materials.size() == 1) ? 0 : materialID;
                if (effectiveMaterialID < 0) {
                    mesh->diffuse = owl::vec3f(1,0,0);
                    mesh->specular = owl::vec3f(0.35f,0.35f,0.35f);
                    mesh->shininess = 64.0f;
                    mesh->diffuseTextureID = -1;
                } 
                else {
                    mesh->diffuse = (const owl::vec3f&)materials[effectiveMaterialID].diffuse;
                    mesh->specular = (const owl::vec3f&)materials[effectiveMaterialID].specular;
                    mesh->shininess = materials[effectiveMaterialID].shininess > 0.0f
                                      ? materials[effectiveMaterialID].shininess
                                      : 64.0f;
                    mesh->diffuseTextureID = loadTexture(model,
                                                         knownTextures,
                                                         textureFile.empty() ? materials[effectiveMaterialID].diffuse_texname : textureFile,
                                                         textureFile.empty() ? materialDir : modelDir);
                }
                if (effectiveMaterialID < 0 && !textureFile.empty()) {
                    mesh->diffuseTextureID = loadTexture(model,
                                                         knownTextures,
                                                         textureFile,
                                                         modelDir);
                }
            }

            if (mesh->vertex.empty())
                delete mesh;
            else {
                // just for sanity's sake:
                if (mesh->texcoord.size() > 0)
                    mesh->texcoord.resize(mesh->vertex.size());
                // just for sanity's sake:
                if (mesh->normal.size() > 0)
                    mesh->normal.resize(mesh->vertex.size());

                for (auto idx : mesh->index) {
                    if (idx.x < 0 || idx.x >= (int)mesh->vertex.size() ||
                        idx.y < 0 || idx.y >= (int)mesh->vertex.size() ||
                        idx.z < 0 || idx.z >= (int)mesh->vertex.size())
                            throw std::runtime_error("invalid triangle indices");
                }
                model->meshes.push_back(mesh);
            }
        }
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
    for (auto vtx : mesh->vertex)
    model->bounds.extend(vtx);

    return model;
}
