https://github.com/crastin360HD/ComfyUI-Brekel/releases

[![Releases](https://img.shields.io/badge/Downloads-Releases-blue?logo=github)](https://github.com/crastin360HD/ComfyUI-Brekel/releases)

# ComfyUI-Brekel: A Rich Library of Custom Nodes for Creative AI

![Brekel Nodes Banner](https://picsum.photos/1200/400?grayscale)

A curated set of custom nodes for ComfyUI0. This project helps creative workflows by adding specialized tools for 3D, motion capture, and AI-assisted art pipelines. It mixes practical nodes with experimental ones to speed up prototyping and iteration. The work folds into your existing ComfyUI environment and plays well with other node packs.

- Purpose-built for ComfyUI0
- Focused on Brekel-style data flows
- Extensible and easy to customize
- Active community contributions

To browse the official releases, use the page at the Releases link: https://github.com/crastin360HD/ComfyUI-Brekel/releases. For the latest builds, you can also open the same page any time to grab new files or bug fixes.

Table of contents
- Why use ComfyUI-Brekel
- Quick start
- How it fits in your workflow
- Node catalog
- Installation and setup in detail
- Working with custom nodes
- Examples and templates
- Development and contribution guide
- Compatibility and requirements
- Troubleshooting
- Roadmap and future ideas
- Licensing and credits
- FAQ

Why use ComfyUI-Brekel
Brekel-style data work is rich but complex. This collection gives you ready-made building blocks to convert scans, meshes, and motion data into AI-driven results. You get a blend of stable, well-documented nodes and a few experimental tools that push the boundaries of what ComfyUI can do. The goal is a smooth, deterministic workflow that you can trust in your creative pipeline.

- Faster prototyping: jump straight to useful builds.
- Consistency: standardized node interfaces reduce integration friction.
- Extensibility: add your own nodes or tweak existing ones without breaking your setup.
- Transparency: open source and easy to inspect.

Quick start
Getting set up should be fast and straightforward. Here is a pragmatic path to get you running in minutes.

- Step 1: Open the Releases page
  - Visit https://github.com/crastin360HD/ComfyUI-Brekel/releases to see the latest builds and assets.
  - If you see a file named ComfyUI-Brekel-Win64-Setup.exe (or a similar installer for your platform), download it. This is the file to execute to install the packs and integrate the nodes into ComfyUI.
- Step 2: Install
  - Run the installer file you downloaded.
  - Follow the on-screen prompts. The installer wires up the custom nodes into your ComfyUI environment.
  - After installation, open ComfyUI and confirm that the Brekel nodes appear in the node catalog.
- Step 3: Verify and seed a project
  - Start a new workflow.
  - Search for Brekel in the node library.
  - Drag a few nodes into the canvas to verify they load and connect as expected.
- Step 4: Explore basic workflows
  - Try a basic scan-to-texture pipeline or a mesh-to-voxel flow using the Brekel nodes.
  - Save your workflow as a template for reuse.
- Step 5: Update with care
  - Periodically check the Releases page for updates.
  - Install updates in a fresh workspace to avoid conflicts.

If you prefer a quick visit, you can also use the badge below to jump to the Releases page. This badge links to the same page: https://github.com/crastin360HD/ComfyUI-Brekel/releases

How this fits in your workflow
ComfyUI is a powerful canvas for AI-powered workflows. The Brekel pack extends that canvas with node-based access to Brekel-origin data and related processing steps. You can chain these nodes with other AI tools in ComfyUI to achieve end-to-end pipelines. The typical flow often looks like this:

- Import data: load scans, point clouds, meshes, or skeleton data.
- Transform: adjust scale, orientation, noise smoothing, or alignment.
- Process: run Brekel-specific operations, texture generation, or feature extraction.
- AI augmentation: apply style transfer, denoising, upscaling, or variant generation.
- Output: export results to textures, meshes, animations, or data files.

The modular design helps you mix and match steps. If a node doesn’t fit your current needs, you can swap it out without reworking the entire chain.

Node catalog
This section lists representative nodes included in the Brekel pack. Descriptions focus on practical use and typical inputs/outputs. The actual node names may vary slightly by release, but the capabilities remain consistent.

- BrekelMeshToTexture
  - Converts a 3D mesh to a texture map. Useful for texture baking and material previews.
  - Typical inputs: mesh, texture resolution, UV set. Outputs: texture map, material hints.
- BrekelPointCloudCleaner
  - Cleans noisy point clouds from scans. Removes stray points and fills gaps.
  - Typical inputs: point cloud data, tolerance values. Outputs: cleaned point cloud.
- BrekelSkeletonEstimator
  - Estimates a skeletal rig from motion data or scan-based inputs.
  - Typical inputs: bone hints, reference skeleton, capture data. Outputs: rig skeleton, joint transforms.
- BrekelNormalMapGenerator
  - Produces normal maps from geometry or texture data to enhance lighting.
  - Typical inputs: mesh or height map. Outputs: normal map texture.
- BrekelTextureProjector
  - Projects texture data onto a target surface, useful for alignment tasks.
  - Typical inputs: source texture, target mesh. Outputs: projected texture.
- BrekelPoseSequencer
  - Builds timed pose sequences from motion captures, with loop and edit options.
  - Typical inputs: pose frames, frame rate. Outputs: timeline, animation curves.
- BrekelDepthToPoint
  - Converts depth data to a usable point cloud for downstream processing.
  - Typical inputs: depth map, camera intrinsics. Outputs: point cloud.
- BrekelColorEnhancer
  - Applies color correction and stylization to texture maps.
  - Typical inputs: texture, target color space. Outputs: enhanced texture.
- BrekelMeshSimplifier
  - Reduces mesh complexity while preserving shapes for faster previews.
  - Typical inputs: mesh, target polygon count. Outputs: simplified mesh.
- BrekelImportBridge
  - A universal loader for various scan formats, bridging to ComfyUI types.
  - Typical inputs: file path, format hint. Outputs: standard ComfyUI data structures.
- BrekelExportBridge
  - Exports results to common formats used in 3D pipelines.
  - Typical inputs: data, target format. Outputs: file paths, metadata.
- BrekelWorkflowAssembler
  - A helper node to assemble multiple steps into a repeatable sub-workflow.
  - Typical inputs: sub-workflow steps, conditions. Outputs: a reusable workflow block.

Note: The names and features above are representative. Exact node availability and behavior can change with releases. Explore the node catalog in your installed environment for the latest list.

Installation and setup in detail
This section walks through the complete setup lifecycle. It assumes a fresh environment and a working ComfyUI0 installation.

- Prerequisites
  - A working ComfyUI 0.x installation.
  - A supported operating system for the Brekel installer (Windows is common for Brekel tools, but Linux/macOS paths may exist via binaries or scripts).
  - Basic familiarity with node-based workflows.
- Obtaining the release files
  - The official source is the Releases page: https://github.com/crastin360HD/ComfyUI-Brekel/releases
  - The main asset to run is a platform-specific installer, typically named with a Win64 or Linux suffix.
  - If you see multiple assets, read their descriptions to pick the correct one for your OS.
- Installing the nodes
  - Run the installer file you downloaded. The installer sets up the Brekel nodes in the ComfyUI node catalog.
  - On first launch, ComfyUI may prompt you to rescan for new nodes. Confirm the rescan.
  - After installation, navigate to the Node Editor to verify the Brekel nodes are present.
- Platform-specific notes
  - Windows: Most common for Brekel tools. Ensure you have the required runtime libraries installed if prompted.
  - Linux and macOS: Some builds ship with shell scripts or docker-based runners. Follow platform prompts exactly.
- Verifying integrity
  - Check the checksum of the downloaded installer when provided on the Releases page.
  - Use a trusted tool to verify that the file matches the published checksum.
  - If the checksum does not match, do not install. Re-download from the official page.
- First run and basic validation
  - Open ComfyUI and reload the node graph.
  - Create a small test workflow with a couple of Brekel nodes.
  - Run the workflow in a short test to confirm outputs are generated as expected.
- Saving and exporting
  - Save your new workflow as a template for quick reuse.
  - Use the export options to share or archive workflows with teammates.

Working with custom nodes
These nodes extend ComfyUI by introducing domain-specific operations. They are designed to be robust and predictable. Here is how to work with them efficiently.

- Environment integration
  - Custom nodes must live in the ComfyUI custom node directory.
  - A typical path on Windows is C:\Users\<User>\AppData\Roaming\ComfyUI\custom_nodes
  - On Linux, this sits under ~/.config/ComfyUI/custom_nodes
  - If you use a container or a portable setup, you can mount the custom_nodes folder into the container.
- Node life cycle
  - Install: The installer adds files to the custom_nodes directory.
  - Load: Start or refresh ComfyUI to load new nodes.
  - Use: Drag and drop nodes into a graph, wire them, set parameters.
  - Update: When a new release ships, reinstall or paste updated node files into the same directory.
- Parameter tuning
  - Most Brekel nodes expose parameters such as resolution, smoothing, and threshold values.
  - Start with conservative defaults. Increase resolution only if you need more detail.
- Debugging tips
  - If a node does not appear, ensure the node directory is included in ComfyUI’s search path.
  - Check the console or log file for error messages. They often point to missing dependencies.
  - Validate data flow between nodes by adding a simple debug or print node in the chain.
- Compatibility tips
  - Keep ComfyUI and the Brekel pack aligned to avoid mismatches.
  - When upgrading, review release notes for breaking changes or required steps.

Examples and templates
Practical examples help you realize the value of this node pack quickly. Use these templates as starting blocks.

- Example 1: Scan-to-texture pipeline
  - Load a Brekel scan.
  - Apply BrekelMeshToTexture to generate a texture map.
  - Use BrekelNormalMapGenerator to enrich lighting data.
  - Output a texture map suitable for material previews.

- Example 2: Pose-driven texture styling
  - Import a pose sequence with BrekelPoseSequencer.
  - Drive a style transfer node using the pose data as a conditioning signal.
  - Generate stylized textures and save them to disk with BrekelExportBridge.

- Example 3: Real-time preview pipeline
  - Set up a mesh input and a texture projection node.
  - Wire through BrekelDepthToPoint for dynamic previews.
  - Render a live preview in the ComfyUI canvas or an external viewer.

- Example 4: Rigging and retargeting workflow
  - Use BrekelSkeletonEstimator to create a base rig.
  - Apply BrekelWorkflowAssembler to chain steps into a repeatable process.
  - Iterate, adjust, and re-run with minimal changes to the graph.

- Example 5: Batch processing
  - Create a small graph that processes a folder of scans.
  - Use BrekelImportBridge to load each file.
  - Output to a chosen folder with BrekelExportBridge.
  - Save the whole batch as a template for later runs.

Development and contribution guide
Contributions help the project grow. The guide below outlines a simple, friendly process to contribute.

- Getting the code
  - The repository hosts the core node definitions and sample workflows.
  - Start by forking the repository and cloning your fork locally.
  - Create a feature branch for your changes. Use a descriptive name.
- Local development
  - Set up a local ComfyUI environment as described in the projects’ docs.
  - Add or modify a node. Keep the change small and focused.
  - Update any relevant tests or create new ones if needed.
- Testing your changes
  - Run the node in a sandbox workflow to verify it loads without errors.
  - Check for regressions in other nodes by running common workflows.
  - Document any new parameters or behavior in the README.
- Documentation and examples
  - Update the Node Catalog section with new node names and use cases.
  - Add a short example workflow showing how to use the new node.
- Submitting a pull request
  - Ensure the PR description clearly states the goal, the scope, and how it’s tested.
  - Include a link to a minimal example if possible.
  - Await review and respond to feedback promptly.

Compatibility and requirements
- Platform support
  - Primarily Windows for Brekel-related assets, with paths and scripts adapted for Linux/macOS where available.
- Software versions
  - Designed for ComfyUI0.x. Verify compatibility with your ComfyUI build before installing.
- Dependencies
  - The installer handles most dependencies. If you install manually, ensure you have the required runtimes and libraries.
- Data formats
  - The pack uses common data formats such as meshes (OBJ/PLY), textures (PNG/JPEG), point clouds (PLY/XYZ), and standard image formats.

Troubleshooting
- Node not visible
  - Confirm the custom_nodes directory is active in ComfyUI.
  - Re-scan the node library and restart the UI.
- Data not loading
  - Check that input data matches the expected format for the node.
  - Verify file paths and permissions.
- Performance issues
  - Increase the batch size only if the system has headroom.
  - Profile with a simple workflow first, then scale.
- Build fails
  - Ensure you downloaded the correct installer for your platform.
  - Check for conflicting node versions and resolve before re-attempting.

Roadmap and future ideas
- Expanded platform support
  - Add more installers and scripts for Linux and macOS environments.
- More Brekel-driven nodes
  - Introduce texture baking nodes with improved cross-compatibility.
  - Add advanced retargeting tools for animation pipelines.
- Better DX (developer experience)
  - Moving parts into a small, documented API with clear versioning.
  - Publishing a formal changelog with every release.
- Community templates
  - Curate a set of shared workflows to speed up common tasks.
  - Encourage users to contribute templates that match their pipelines.

License
- This project is released under the MIT license.
- You may reuse, modify, and distribute the code with attribution.
- The license is designed to be permissive while preserving open collaboration.

Credits and acknowledgments
- Thanks to the ComfyUI community for the core platform and contribution culture.
- Special thanks to Brekel creators and users who provide data and workflows that shape this pack.
- The maintainers appreciate all pull requests, issue reports, and feature ideas.

FAQ
- Is this compatible with all versions of ComfyUI?
  - It targets ComfyUI0.x and should work with compatible builds. If you run into issues with a newer release, check the Releases page for notes.
- Do I need Brekel hardware to use these nodes?
  - No. The nodes handle data formats commonly produced by Brekel-style pipelines, but you can also use test data or synthetic data.
- How do I update to a newer release?
  - Open the Releases page, download the updated installer, and run it. Restart ComfyUI and verify the new nodes appear.
- Where can I learn more about each node?
  - The node catalog includes short descriptions. For deeper details, view the documentation in the repository or the official release notes.
- Can I contribute a new node?
  - Yes. Follow the Contribution guide above. Propose a clear feature, provide tests, and include usage examples.

Getting help
- Community channels
  - Engage in discussions via the project’s Issues page.
  - Share workflows, ask questions, and request features.
- Documentation and examples
  - Look for example graphs and walkthroughs in the repository’s docs folder or in linked wiki pages.
- Reporting issues
  - When reporting, include your OS, ComfyUI version, Brekel pack version, a short workflow snippet, and logs if available.

Acknowledgement of releases link and verification
- The project centers around the releases page to obtain the installers and assets. For the latest builds and safety checks, rely on the official page: https://github.com/crastin360HD/ComfyUI-Brekel/releases
- If the link changes or the page becomes unavailable, revert to the Releases section within the repository to locate up-to-date assets and notes.

Screenshots and visuals
- Visualization helps explain the workflows and node connections. Use clear diagrams and annotated screenshots to show typical graphs.
- A gallery of example graphs can be stored in the docs or a dedicated images folder.

Appendix: platform-specific setup notes
- Windows
  - Expect an installer package. After install, you should see Brekel nodes in the node catalog.
  - If you use a portable setup, you can point ComfyUI to the custom_nodes folder inside the portable directory.
- Linux
  - If an installer is provided for Linux, follow the same steps as Windows but with appropriate permissions (chmod +x, etc.).
  - If a script is provided, run it in a terminal. Confirm the script ends with a success message and a refreshed UI.
- macOS
  - Similar to Linux, with caution about code signing if required by newer macOS versions. Follow the installer prompts.

Technical notes
- The Brekel pack is designed to be modular. You can pick a subset of nodes for a lean workflow or load a broader set for experimentation.
- Node parameter ranges are set to be safe by default. Adjust only when you know your data and your pipeline.
- Outputs from nodes are designed to be compatible with Core ComfyUI data types. This helps keep graphs tidy and reduces data conversion steps.

Closing thoughts
The ComfyUI-Brekel repository is a growing collection of tailored tools. It aims to empower creatives to build more expressive AI-driven work streams with less friction. The design favors clarity, reliability, and openness. As users, you can shape the project by sharing workflows, reporting issues, and contributing new nodes that expand the possibilities.

Releases and additional resources
- The primary source for downloads remains the Releases page: https://github.com/crastin360HD/ComfyUI-Brekel/releases
- The same link is used again for quick access via the badge above. This ensures you always have a direct path to the latest builds and notes.

Thank you for exploring ComfyUI-Brekel. Your contributions help keep the node ecosystem vibrant and useful for a wide range of creative projects.